"""
Database connection and initialization for MLX-GUI.
"""

import os
from pathlib import Path
from typing import Generator, Optional, Dict, Any
import appdirs
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
import threading
import time
from functools import lru_cache
import logging

from mlx_gui.models import Base, AppSettings

logger = logging.getLogger(__name__)

# Global cache for settings to reduce database hits
_settings_cache: Dict[str, Any] = {}
_settings_cache_lock = threading.RLock()
_settings_cache_timestamp = 0
_settings_cache_ttl = 30  # 30 seconds TTL


class DatabaseManager:
    """Manages database connections and initialization."""
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_path: Path to SQLite database file. If None, uses default location.
        """
        if database_path is None:
            # Use standard user application directory
            app_dir = appdirs.user_data_dir("mlx-gui", "mlx-gui")
            os.makedirs(app_dir, exist_ok=True)
            database_path = os.path.join(app_dir, "mlx-gui.db")
        
        self.database_path = database_path
        self.database_url = f"sqlite:///{database_path}"
        
        # Create engine with optimized SQLite settings
        self.engine = create_engine(
            self.database_url,
            echo=False,  # Set to True for SQL debugging
            connect_args={
                "check_same_thread": False,  # Allow multiple threads
                "timeout": 30,  # Connection timeout
            },
            # Add connection pooling for better performance
            poolclass=QueuePool,
            pool_size=10,  # Number of connections to maintain
            max_overflow=20,  # Additional connections when pool is full
            pool_pre_ping=True,  # Validate connections before use
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Initialize database
        self._initialize_database()
        
        # Setup connection event listeners for performance monitoring
        self._setup_connection_events()
    
    def _setup_connection_events(self):
        """Setup database connection event listeners for performance monitoring."""
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance."""
            cursor = dbapi_connection.cursor()
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys=ON")
            # Use memory for temporary storage
            cursor.execute("PRAGMA temp_store=MEMORY")
            # Normal synchronous mode (good balance of safety and performance)
            cursor.execute("PRAGMA synchronous=NORMAL")
            # Increase cache size to 32MB for better performance
            cursor.execute("PRAGMA cache_size=32768")
            # Use memory-mapped I/O for better performance
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
            cursor.close()
    
    def _initialize_database(self):
        """Initialize database tables and default settings."""
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        
        # Insert default settings if they don't exist
        self._insert_default_settings()
        
        # Reset all model statuses to unloaded on startup
        self._reset_model_statuses()
        
        # Create additional indexes for better performance
        self._create_performance_indexes()
    
    def _create_performance_indexes(self):
        """Create additional indexes for better query performance."""
        try:
            with self.engine.connect() as connection:
                # Composite index for model queries
                connection.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_models_status_type 
                    ON models(status, model_type)
                """))
                
                # Index for request queue processing
                connection.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_request_queue_model_status_priority 
                    ON request_queue(model_id, status, priority DESC, created_at)
                """))
                
                # Index for inference requests by model and time
                connection.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_inference_requests_model_time 
                    ON inference_requests(model_id, created_at DESC)
                """))
                
                connection.commit()
                logger.info("Created performance indexes")
        except Exception as e:
            logger.warning(f"Could not create performance indexes: {e}")
    
    def _insert_default_settings(self):
        """Insert default application settings."""
        default_settings = [
            ("server_port", 8000, "Default server port"),
            ("max_concurrent_requests", 5, "Maximum concurrent inference requests"),
            ("max_concurrent_requests_per_model", 1, "Maximum concurrent requests per model"),
            ("max_concurrent_models", 3, "Maximum concurrent loaded models"),
            ("auto_unload_inactive_models", True, "Automatically unload models after inactivity"),
            ("model_inactivity_timeout_minutes", 5, "Minutes before unloading inactive models"),
            ("enable_system_tray", True, "Enable system tray integration"),
            ("log_level", "INFO", "Application logging level"),
            ("huggingface_cache_dir", "", "HuggingFace cache directory path"),
            ("enable_gpu_acceleration", True, "Enable GPU acceleration when available"),
            ("bind_to_all_interfaces", False, "Bind server to all interfaces (0.0.0.0) instead of localhost only"),
        ]
        
        with self.get_session() as session:
            for key, value, description in default_settings:
                existing = session.query(AppSettings).filter_by(key=key).first()
                if not existing:
                    setting = AppSettings(key=key, description=description)
                    setting.set_typed_value(value)
                    session.add(setting)
            session.commit()
    
    def _reset_model_statuses(self):
        """Reset all model statuses to unloaded on startup."""
        try:
            with self.get_session() as session:
                from mlx_gui.models import Model
                # Use bulk update for better performance
                session.query(Model).filter(Model.status != "unloaded").update({
                    "status": "unloaded",
                    "last_unloaded_at": None
                })
                session.commit()
                print("🔄 Reset all models to unloaded status on startup")
        except Exception as e:
            # Don't crash on startup if this fails
            print(f"Warning: Could not reset model statuses: {e}")
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def get_session_generator(self) -> Generator[Session, None, None]:
        """Get a database session generator (for dependency injection)."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def close(self):
        """Close database connections."""
        self.engine.dispose()
    
    def get_setting(self, key: str, default=None):
        """Get a setting value by key with caching."""
        global _settings_cache, _settings_cache_timestamp
        
        current_time = time.time()
        
        # Check if cache is still valid
        with _settings_cache_lock:
            if current_time - _settings_cache_timestamp < _settings_cache_ttl:
                if key in _settings_cache:
                    return _settings_cache[key]
            else:
                # Cache expired, clear it
                _settings_cache.clear()
                _settings_cache_timestamp = current_time
        
        # Fetch from database
        with self.get_session() as session:
            setting = session.query(AppSettings).filter_by(key=key).first()
            if setting:
                value = setting.get_typed_value()
                # Update cache
                with _settings_cache_lock:
                    _settings_cache[key] = value
                return value
            return default
    
    def set_setting(self, key: str, value, description: Optional[str] = None):
        """Set a setting value by key and invalidate cache."""
        with self.get_session() as session:
            setting = session.query(AppSettings).filter_by(key=key).first()
            if setting:
                setting.set_typed_value(value)
                if description:
                    setting.description = description
            else:
                setting = AppSettings(key=key, description=description or "")
                setting.set_typed_value(value)
                session.add(setting)
            session.commit()
        
        # Invalidate cache
        global _settings_cache
        with _settings_cache_lock:
            _settings_cache.clear()
    
    def vacuum_database(self):
        """Vacuum the database to reclaim space."""
        with self.engine.connect() as connection:
            connection.execute(text("VACUUM"))
            connection.commit()
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database."""
        import shutil
        shutil.copy2(self.database_path, backup_path)
    
    @lru_cache(maxsize=128)
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path with caching for better performance."""
        if model_path.startswith("~"):
            return os.path.expanduser(model_path)
        
        # Handle HuggingFace cache paths
        if "huggingface" in model_path.lower() or "/.cache/huggingface/" in model_path:
            # This is already a resolved path
            return model_path
        
        # Check if it's a HuggingFace model ID
        if "/" in model_path and not os.path.exists(model_path):
            try:
                from huggingface_hub import snapshot_download
                return snapshot_download(repo_id=model_path, local_files_only=True)
            except Exception:
                pass
        
        return model_path
    
    def update_model_sizes_from_disk(self):
        """Update model memory requirements based on actual file sizes with caching."""
        try:
            with self.get_session() as session:
                from mlx_gui.models import Model
                import os
                
                # Use more efficient query with specific columns
                models = session.query(Model.id, Model.name, Model.path, Model.memory_required_gb).all()
                updated_count = 0
                
                for model_id, model_name, model_path, current_memory in models:
                    if not model_path:
                        continue
                        
                    # Resolve the actual path (handle HuggingFace cache paths)
                    actual_path = self._resolve_model_path(model_path)
                    
                    if os.path.exists(actual_path):
                        # Calculate actual file size with caching
                        new_memory_gb = self._calculate_model_memory_cached(actual_path, model_path)
                        
                        # Update if significantly different (more than 0.5GB difference)
                        if abs(current_memory - new_memory_gb) > 0.5:
                            session.query(Model).filter(Model.id == model_id).update({
                                "memory_required_gb": new_memory_gb
                            })
                            updated_count += 1
                            print(f"📊 Updated {model_name}: {current_memory:.1f}GB → {new_memory_gb:.1f}GB")
                    else:
                        print(f"⚠️  Model path not found: {model_name} -> {actual_path}")
                
                session.commit()
                
                if updated_count > 0:
                    print(f"✅ Updated memory requirements for {updated_count} models")
                else:
                    print("ℹ️  No models needed size updates")
                    
        except Exception as e:
            print(f"Error updating model sizes: {e}")
    
    @lru_cache(maxsize=256)
    def _calculate_model_memory_cached(self, actual_path: str, model_path: str) -> float:
        """Calculate model memory with caching for better performance."""
        total_size_bytes = 0
        
        try:
            # Walk through all files in the model directory
            for root, dirs, files in os.walk(actual_path):
                for file in files:
                    # Count all model-related files
                    if file.endswith(('.safetensors', '.bin', '.pth', '.pt', '.gguf', '.npz')):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size_bytes += os.path.getsize(file_path)
            
            # Convert to GB
            file_size_gb = total_size_bytes / (1024**3)
            
            # Add MLX overhead (25% for inference, activations, etc.)
            # Audio models might need less overhead, text models might need more
            if "whisper" in model_path.lower() or "parakeet" in model_path.lower():
                overhead_multiplier = 1.15  # 15% overhead for audio models
            else:
                overhead_multiplier = 1.25  # 25% overhead for text models
                
            actual_memory_gb = file_size_gb * overhead_multiplier
            
            # Round to one decimal place
            actual_memory_gb = round(actual_memory_gb, 1)
            
            return max(actual_memory_gb, 0.1)  # Minimum 0.1GB
            
        except Exception as e:
            logger.warning(f"Could not calculate actual memory usage for {model_path}: {e}")
            # Fallback to a reasonable default
            return 2.0
    
    def get_database_size(self) -> int:
        """Get database file size in bytes."""
        return os.path.getsize(self.database_path)
    
    def get_database_info(self) -> dict:
        """Get database information and statistics."""
        try:
            with self.engine.connect() as connection:
                # Get table sizes
                result = connection.execute(text("""
                    SELECT name, sql FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """))
                tables = result.fetchall()
                
                table_info = {}
                for table_name, _ in tables:
                    # Get row count
                    count_result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar()
                    
                    # Get table size
                    size_result = connection.execute(text(f"""
                        SELECT SUM(length(hex(length(quote(t.*)))) + length(quote(t.*))) 
                        FROM {table_name} t
                    """))
                    table_size = size_result.scalar() or 0
                    
                    table_info[table_name] = {
                        "row_count": row_count,
                        "size_bytes": table_size
                    }
                
                return {
                    "database_path": self.database_path,
                    "file_size_bytes": self.get_database_size(),
                    "tables": table_info,
                    "connection_pool_size": getattr(self.engine.pool, 'size', lambda: 0)(),
                    "connection_pool_checked_in": getattr(self.engine.pool, 'checkedin', lambda: 0)(),
                    "connection_pool_checked_out": getattr(self.engine.pool, 'checkedout', lambda: 0)(),
                }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None
_database_manager_lock = threading.Lock()


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _database_manager
    
    if _database_manager is None:
        with _database_manager_lock:
            if _database_manager is None:
                _database_manager = DatabaseManager()
    
    return _database_manager


def get_db_session() -> Generator[Session, None, None]:
    """Get a database session generator for dependency injection."""
    db_manager = get_database_manager()
    yield from db_manager.get_session_generator()


def close_database():
    """Close the global database manager."""
    global _database_manager
    if _database_manager:
        _database_manager.close()
        _database_manager = None