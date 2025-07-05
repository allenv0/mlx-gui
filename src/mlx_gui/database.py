"""
Database connection and initialization for MLX-GUI.
"""

import os
from pathlib import Path
from typing import Generator, Optional
import appdirs
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from mlx_gui.models import Base, AppSettings


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
        
        # Create engine with SQLite optimizations
        self.engine = create_engine(
            self.database_url,
            echo=False,  # Set to True for SQL debugging
            connect_args={
                "check_same_thread": False,  # Allow multiple threads
                "timeout": 30,  # Connection timeout
            }
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables and default settings."""
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        
        # Enable WAL mode for better concurrency
        with self.engine.connect() as connection:
            connection.execute(text("PRAGMA journal_mode=WAL"))
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.execute(text("PRAGMA temp_store=MEMORY"))
            connection.execute(text("PRAGMA synchronous=NORMAL"))
            connection.commit()
        
        # Insert default settings if they don't exist
        self._insert_default_settings()
    
    def _insert_default_settings(self):
        """Insert default application settings."""
        default_settings = [
            ("server_port", 8000, "Default server port"),
            ("max_concurrent_requests", 5, "Maximum concurrent inference requests"),
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
        """Get a setting value by key."""
        with self.get_session() as session:
            setting = session.query(AppSettings).filter_by(key=key).first()
            if setting:
                return setting.get_typed_value()
            return default
    
    def set_setting(self, key: str, value, description: str = None):
        """Set a setting value by key."""
        with self.get_session() as session:
            setting = session.query(AppSettings).filter_by(key=key).first()
            if setting:
                setting.set_typed_value(value)
                if description:
                    setting.description = description
            else:
                setting = AppSettings(key=key, description=description)
                setting.set_typed_value(value)
                session.add(setting)
            session.commit()
    
    def vacuum_database(self):
        """Vacuum the database to reclaim space."""
        with self.engine.connect() as connection:
            connection.execute(text("VACUUM"))
            connection.commit()
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database."""
        import shutil
        shutil.copy2(self.database_path, backup_path)
    
    def get_database_size(self) -> int:
        """Get database file size in bytes."""
        return os.path.getsize(self.database_path)
    
    def get_database_info(self) -> dict:
        """Get database information."""
        with self.engine.connect() as connection:
            result = connection.execute(text("PRAGMA database_list")).fetchall()
            main_db = next((row for row in result if row[1] == "main"), None)
            
            if main_db:
                return {
                    "path": main_db[2],
                    "size_bytes": self.get_database_size(),
                    "journal_mode": connection.execute(text("PRAGMA journal_mode")).scalar(),
                    "foreign_keys": bool(connection.execute(text("PRAGMA foreign_keys")).scalar()),
                    "synchronous": connection.execute(text("PRAGMA synchronous")).scalar(),
                }
        return {}


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def get_db_session() -> Generator[Session, None, None]:
    """Dependency function to get database session (for FastAPI)."""
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def close_database():
    """Close the global database connection."""
    global db_manager
    if db_manager:
        db_manager.close()
        db_manager = None