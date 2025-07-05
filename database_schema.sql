-- MLX-GUI Database Schema
-- SQLite database for managing MLX models and application state

-- Models table - stores information about loaded/available models
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    version TEXT,
    model_type TEXT NOT NULL, -- 'text', 'vision', 'audio', 'multimodal'
    huggingface_id TEXT,
    memory_required_gb INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'unloaded', -- 'loaded', 'unloaded', 'loading', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    use_count INTEGER DEFAULT 0,
    error_message TEXT,
    metadata TEXT -- JSON string for additional model metadata
);

-- Model capabilities table - tracks what each model can do
CREATE TABLE model_capabilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    capability TEXT NOT NULL, -- 'text_generation', 'image_understanding', 'audio_processing', etc.
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
    UNIQUE(model_id, capability)
);

-- Inference sessions table - tracks active inference sessions
CREATE TABLE inference_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    model_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active', -- 'active', 'completed', 'failed'
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

-- Inference requests table - logs all inference requests
CREATE TABLE inference_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    model_id INTEGER NOT NULL,
    request_type TEXT NOT NULL, -- 'text', 'multimodal', 'audio'
    input_data TEXT NOT NULL, -- JSON string containing input data
    output_data TEXT, -- JSON string containing output data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    error_message TEXT,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

-- System metrics table - tracks system performance and resource usage
CREATE TABLE system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    memory_used_gb REAL NOT NULL,
    memory_total_gb REAL NOT NULL,
    cpu_usage_percent REAL,
    gpu_memory_used_gb REAL,
    gpu_memory_total_gb REAL,
    active_models_count INTEGER DEFAULT 0
);

-- Application settings table - stores user preferences and configuration
CREATE TABLE app_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    value_type TEXT NOT NULL DEFAULT 'string', -- 'string', 'integer', 'boolean', 'json'
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Queue table - manages inference request queue for multi-user support
CREATE TABLE request_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    model_id INTEGER NOT NULL,
    request_data TEXT NOT NULL, -- JSON string
    priority INTEGER DEFAULT 0, -- higher number = higher priority
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'queued', -- 'queued', 'processing', 'completed', 'failed'
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_last_used ON models(last_used_at);
CREATE INDEX idx_inference_sessions_model_id ON inference_sessions(model_id);
CREATE INDEX idx_inference_requests_session_id ON inference_requests(session_id);
CREATE INDEX idx_inference_requests_model_id ON inference_requests(model_id);
CREATE INDEX idx_inference_requests_created_at ON inference_requests(created_at);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX idx_request_queue_status ON request_queue(status);
CREATE INDEX idx_request_queue_priority ON request_queue(priority);

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_models_timestamp 
    AFTER UPDATE ON models
    FOR EACH ROW
    BEGIN
        UPDATE models SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER update_settings_timestamp 
    AFTER UPDATE ON app_settings
    FOR EACH ROW
    BEGIN
        UPDATE app_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

-- Insert default application settings
INSERT INTO app_settings (key, value, value_type, description) VALUES
    ('server_port', '8000', 'integer', 'Default server port'),
    ('max_concurrent_requests', '5', 'integer', 'Maximum concurrent inference requests'),
    ('auto_unload_inactive_models', 'true', 'boolean', 'Automatically unload models after inactivity'),
    ('model_inactivity_timeout_minutes', '30', 'integer', 'Minutes before unloading inactive models'),
    ('enable_system_tray', 'true', 'boolean', 'Enable system tray integration'),
    ('log_level', 'INFO', 'string', 'Application logging level'),
    ('huggingface_cache_dir', '', 'string', 'HuggingFace cache directory path'),
    ('enable_gpu_acceleration', 'true', 'boolean', 'Enable GPU acceleration when available');