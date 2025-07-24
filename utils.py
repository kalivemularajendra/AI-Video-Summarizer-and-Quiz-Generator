"""
Configuration utilities for the Video Analysis & Quiz Generator
"""
import configparser
import os
from pathlib import Path
from typing import Any, List

class AppConfig:
    """Application configuration manager"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        config_path = Path(__file__).parent / self.config_file
        if config_path.exists():
            self.config.read(config_path)
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration"""
        self.config['app'] = {
            'max_file_size_mb': '500',
            'supported_formats': 'mp4,avi,mov,mkv,webm,flv,m4v',
            'session_timeout_minutes': '30',
            'max_retry_attempts': '3',
            'cache_ttl_seconds': '3600'
        }
        
        self.config['performance'] = {
            'enable_caching': 'true',
            'enable_progress_tracking': 'true',
            'enable_background_processing': 'false'
        }
        
        self.config['security'] = {
            'validate_api_keys': 'true',
            'log_sensitive_info': 'false'
        }
        
        self.config['api'] = {
            'gemini_model': 'gemini-2.0-flash-lite',
            'groq_model': 'llama-3.3-70b-versatile',
            'request_timeout': '300'
        }
        
        self.config['ui'] = {
            'page_layout': 'wide',
            'show_usage_tips': 'false',
            'enable_dark_mode': 'false',
            'show_system_status': 'true'
        }
        
        self.config['logging'] = {
            'log_level': 'INFO',
            'log_file_path': '',
            'enable_detailed_errors': 'true'
        }
        
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        config_path = Path(__file__).parent / self.config_file
        with open(config_path, 'w') as f:
            self.config.write(f)
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get configuration value"""
        try:
            return self.config.get(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        """Get integer configuration value"""
        try:
            return self.config.getint(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def get_bool(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean configuration value"""
        try:
            return self.config.getboolean(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def get_list(self, section: str, key: str, fallback: List[str] = None) -> List[str]:
        """Get list configuration value"""
        if fallback is None:
            fallback = []
        
        try:
            value = self.config.get(section, key)
            return [item.strip() for item in value.split(',') if item.strip()]
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    # Convenience methods for common configurations
    @property
    def max_file_size_mb(self) -> int:
        return self.get_int('app', 'max_file_size_mb', 500)
    
    @property
    def supported_formats(self) -> List[str]:
        return self.get_list('app', 'supported_formats', ['mp4', 'avi', 'mov', 'mkv', 'webm'])
    
    @property
    def session_timeout_minutes(self) -> int:
        return self.get_int('app', 'session_timeout_minutes', 30)
    
    @property
    def max_retry_attempts(self) -> int:
        return self.get_int('app', 'max_retry_attempts', 3)
    
    @property
    def cache_ttl_seconds(self) -> int:
        return self.get_int('app', 'cache_ttl_seconds', 3600)
    
    @property
    def enable_caching(self) -> bool:
        return self.get_bool('performance', 'enable_caching', True)
    
    @property
    def validate_api_keys(self) -> bool:
        return self.get_bool('security', 'validate_api_keys', True)
    
    @property
    def gemini_model(self) -> str:
        return self.get('api', 'gemini_model', 'gemini-2.0-flash-lite')
    
    @property
    def groq_model(self) -> str:
        return self.get('api', 'groq_model', 'llama-3.3-70b-versatile')
    
    @property
    def request_timeout(self) -> int:
        return self.get_int('api', 'request_timeout', 300)
    
    @property
    def page_layout(self) -> str:
        return self.get('ui', 'page_layout', 'wide')
    
    @property
    def show_usage_tips(self) -> bool:
        return self.get_bool('ui', 'show_usage_tips', False)
    
    @property
    def show_system_status(self) -> bool:
        return self.get_bool('ui', 'show_system_status', True)
    
    @property
    def log_level(self) -> str:
        return self.get('logging', 'log_level', 'INFO')
    
    @property
    def enable_detailed_errors(self) -> bool:
        return self.get_bool('logging', 'enable_detailed_errors', True)

# Global configuration instance
config = AppConfig()
