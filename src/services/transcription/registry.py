"""
Connector registry for managing transcription connectors.

Provides factory pattern for creating and managing transcription connectors,
with auto-detection from environment variables for backwards compatibility.
"""

import os
import logging
from typing import Dict, Any, Optional, Type, List

from .base import BaseTranscriptionConnector, TranscriptionCapability, TranscriptionRequest, TranscriptionResponse
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConnectorRegistry:
    """
    Registry for managing transcription connectors.

    Singleton pattern - use get_registry() to get the shared instance.
    """

    _instance = None
    _connectors: Dict[str, Type[BaseTranscriptionConnector]] = {}
    _active_connector: Optional[BaseTranscriptionConnector] = None
    _connector_name: str = ""
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._register_builtin_connectors()
        self._initialized = True

    def _register_builtin_connectors(self):
        """Register all built-in connectors."""
        from .connectors.openai_whisper import OpenAIWhisperConnector
        from .connectors.openai_transcribe import OpenAITranscribeConnector
        from .connectors.asr_endpoint import ASREndpointConnector
        from .connectors.azure_openai_transcribe import AzureOpenAITranscribeConnector
        from .connectors.mistral import MistralTranscriptionConnector
        from .connectors.vibevoice import VibeVoiceTranscriptionConnector
        from .connectors.alibaba_funasr import AlibabaFunASRConnector

        self.register('openai_whisper', OpenAIWhisperConnector)
        self.register('openai_transcribe', OpenAITranscribeConnector)
        self.register('asr_endpoint', ASREndpointConnector)
        self.register('azure_openai_transcribe', AzureOpenAITranscribeConnector)
        self.register('mistral', MistralTranscriptionConnector)
        self.register('vibevoice', VibeVoiceTranscriptionConnector)
        self.register('alibaba_funasr', AlibabaFunASRConnector)

    def register(self, name: str, connector_class: Type[BaseTranscriptionConnector]):
        """
        Register a connector class.

        Args:
            name: Unique name for the connector
            connector_class: The connector class to register
        """
        self._connectors[name] = connector_class
        logger.debug(f"Registered transcription connector: {name}")

    def get_connector_class(self, name: str) -> Type[BaseTranscriptionConnector]:
        """
        Get a connector class by name.

        Args:
            name: The connector name

        Returns:
            The connector class

        Raises:
            ConfigurationError: If connector not found
        """
        if name not in self._connectors:
            raise ConfigurationError(
                f"Unknown connector: {name}. Available: {list(self._connectors.keys())}"
            )
        return self._connectors[name]

    def create_connector(self, name: str, config: Dict[str, Any]) -> BaseTranscriptionConnector:
        """
        Create a connector instance.

        Args:
            name: The connector name
            config: Configuration dict for the connector

        Returns:
            Configured connector instance
        """
        connector_class = self.get_connector_class(name)
        return connector_class(config)

    def list_connectors(self) -> List[Dict[str, Any]]:
        """
        List all registered connectors with their capabilities.

        Returns:
            List of connector info dicts
        """
        result = []
        for name, cls in self._connectors.items():
            result.append({
                'name': name,
                'provider_name': cls.PROVIDER_NAME,
                'capabilities': [c.name for c in cls.CAPABILITIES],
                'config_schema': cls.get_config_schema()
            })
        return result

    def initialize_from_env(self) -> BaseTranscriptionConnector:
        """
        Initialize the active connector from environment variables.

        Auto-detection priority:
        1. TRANSCRIPTION_CONNECTOR - explicit connector name
        2. ASR_BASE_URL contains 'dashscope.aliyuncs.com' AND ENABLE_UPLOAD_BUCKET=true - use Alibaba FunASR
        3. ASR_BASE_URL is set - use ASR endpoint (smarter detection)
           - USE_ASR_ENDPOINT=true also works (backwards compat, with deprecation warning)
        4. TRANSCRIPTION_MODEL contains 'gpt-4o' - use OpenAI Transcribe
        5. TRANSCRIPTION_MODEL is set - use OpenAI Whisper with that model
        6. Default to OpenAI Whisper (whisper-1)

        Returns:
            The initialized connector
        """
        connector_name = os.environ.get('TRANSCRIPTION_CONNECTOR', '').lower().strip()

        if not connector_name:
            # Auto-detect based on existing config for backwards compatibility
            asr_base_url = os.environ.get('ASR_BASE_URL', '').strip()
            use_asr_flag = os.environ.get('USE_ASR_ENDPOINT', 'false').lower() == 'true'
            transcription_model = os.environ.get('TRANSCRIPTION_MODEL', '').lower()
            whisper_model = os.environ.get('WHISPER_MODEL', '').lower()
            enable_upload_bucket = os.environ.get('ENABLE_UPLOAD_BUCKET', 'false').lower() == 'true'

            # Deprecation warning for legacy USE_ASR_ENDPOINT flag
            if use_asr_flag:
                logger.warning(
                    "USE_ASR_ENDPOINT=true is deprecated. "
                    "Set ASR_BASE_URL instead for auto-detection, or use TRANSCRIPTION_CONNECTOR=asr_endpoint"
                )

            # Priority 2: Alibaba FunASR - check if ASR_BASE_URL contains dashscope.aliyuncs.com AND ENABLE_UPLOAD_BUCKET=true
            # Only use Alibaba FunASR when bucket upload is enabled (requires public URLs)
            if asr_base_url and 'dashscope.aliyuncs.com' in asr_base_url and enable_upload_bucket:
                connector_name = 'alibaba_funasr'
                logger.info("Auto-detected Alibaba FunASR from ASR_BASE_URL (ENABLE_UPLOAD_BUCKET=true)")
            # Priority 3: ASR endpoint - check ASR_BASE_URL or legacy flag
            elif asr_base_url or use_asr_flag:
                connector_name = 'asr_endpoint'
                if asr_base_url:
                    logger.info("Auto-detected ASR endpoint from ASR_BASE_URL")
            # Priority 3.5: Azure OpenAI - check for Azure endpoint URL
            elif self._is_azure_endpoint():
                connector_name = 'azure_openai_transcribe'
                logger.info("Auto-detected Azure OpenAI from TRANSCRIPTION_BASE_URL")
            # Priority 4: Model-based detection
            elif transcription_model and 'gpt-4o' in transcription_model:
                connector_name = 'openai_transcribe'
                logger.info(f"Auto-detected OpenAI Transcribe from TRANSCRIPTION_MODEL={transcription_model}")
            # Priority 5 & 6: OpenAI Whisper (with custom or default model)
            else:
                connector_name = 'openai_whisper'
                model = transcription_model or whisper_model or 'whisper-1'
                logger.info(f"Using OpenAI Whisper connector with model: {model}")

        config = self._build_config_from_env(connector_name)

        try:
            self._active_connector = self.create_connector(connector_name, config)
            self._connector_name = connector_name

            logger.info(f"Initialized transcription connector: {connector_name}")
            logger.info(f"Capabilities: {[c.name for c in self._active_connector.get_capabilities()]}")

            return self._active_connector

        except Exception as e:
            logger.error(f"Failed to initialize connector '{connector_name}': {e}")
            raise ConfigurationError(f"Failed to initialize connector '{connector_name}': {e}") from e

    def _get_asr_timeout(self) -> int:
        """
        Get ASR timeout with fallback chain: ENV -> Admin UI -> default.

        Priority:
        1. ASR_TIMEOUT environment variable
        2. asr_timeout_seconds environment variable (legacy)
        3. SystemSetting from Admin UI (database)
        4. Default: 1800 seconds (30 minutes)
        """
        # Check environment variables first
        env_timeout = os.environ.get('ASR_TIMEOUT') or os.environ.get('asr_timeout_seconds')
        if env_timeout:
            return int(env_timeout)

        # Fall back to Admin UI setting (SystemSetting in database)
        try:
            from src.models import SystemSetting
            db_timeout = SystemSetting.get_setting('asr_timeout_seconds', None)
            if db_timeout is not None:
                return int(db_timeout)
        except Exception as e:
            # May fail if no app context or during initialization
            logger.debug(f"Could not read ASR timeout from database: {e}")

        # Default: 30 minutes
        return 1800

    def _is_azure_endpoint(self) -> bool:
        """Check if the TRANSCRIPTION_BASE_URL points to an Azure OpenAI endpoint."""
        base_url = os.environ.get('TRANSCRIPTION_BASE_URL', '').lower()
        return '.openai.azure.com' in base_url or '.cognitiveservices.azure.com' in base_url

    def _build_config_from_env(self, connector_name: str) -> Dict[str, Any]:
        """
        Build connector config from environment variables.

        Args:
            connector_name: The connector to build config for

        Returns:
            Configuration dict
        """
        if connector_name == 'asr_endpoint':
            base_url = os.environ.get('ASR_BASE_URL', '')
            if base_url:
                base_url = base_url.split('#')[0].strip()

            return {
                'base_url': base_url,
                'timeout': self._get_asr_timeout(),
                'diarize': os.environ.get('ASR_DIARIZE', 'true').lower() == 'true',
                'return_speaker_embeddings': os.environ.get('ASR_RETURN_SPEAKER_EMBEDDINGS', 'false').lower() == 'true'
            }

        elif connector_name == 'openai_transcribe':
            base_url = os.environ.get('TRANSCRIPTION_BASE_URL', 'https://api.openai.com/v1')
            if base_url:
                base_url = base_url.split('#')[0].strip()

            return {
                'api_key': os.environ.get('TRANSCRIPTION_API_KEY', ''),
                'base_url': base_url,
                'model': os.environ.get('TRANSCRIPTION_MODEL', 'gpt-4o-transcribe')
            }

        elif connector_name == 'azure_openai_transcribe':
            # Azure OpenAI requires endpoint and deployment_name
            # TRANSCRIPTION_BASE_URL should be the Azure endpoint (e.g., https://your-resource.openai.azure.com)
            endpoint = os.environ.get('TRANSCRIPTION_BASE_URL', '')
            if endpoint:
                endpoint = endpoint.split('#')[0].strip()
                # Remove any trailing /openai or /v1 paths - we build the full URL ourselves
                endpoint = endpoint.rstrip('/')
                for suffix in ['/openai/v1', '/openai', '/v1']:
                    if endpoint.lower().endswith(suffix):
                        endpoint = endpoint[:-len(suffix)]

            return {
                'api_key': os.environ.get('TRANSCRIPTION_API_KEY', ''),
                'endpoint': endpoint,
                'deployment_name': os.environ.get('AZURE_DEPLOYMENT_NAME', os.environ.get('TRANSCRIPTION_MODEL', 'gpt-4o-transcribe')),
                'api_version': os.environ.get('AZURE_API_VERSION', '2025-04-01-preview'),
                'model': os.environ.get('TRANSCRIPTION_MODEL', '')  # For capability detection
            }

        elif connector_name == 'mistral':
            base_url = os.environ.get('TRANSCRIPTION_BASE_URL', '')
            if base_url:
                base_url = base_url.split('#')[0].strip()

            return {
                'api_key': os.environ.get('TRANSCRIPTION_API_KEY', ''),
                'base_url': base_url or 'https://api.mistral.ai',
                'model': os.environ.get('TRANSCRIPTION_MODEL', 'voxtral-mini-latest'),
            }

        elif connector_name == 'vibevoice':
            base_url = os.environ.get('TRANSCRIPTION_BASE_URL', '')
            if base_url:
                base_url = base_url.split('#')[0].strip()

            return {
                'base_url': base_url,
                'model': os.environ.get('TRANSCRIPTION_MODEL', 'microsoft/VibeVoice-ASR'),
                'api_key': os.environ.get('TRANSCRIPTION_API_KEY', ''),
            }

        elif connector_name == 'alibaba_funasr':
            base_url = os.environ.get('ASR_BASE_URL', '')
            if base_url:
                base_url = base_url.split('#')[0].strip()

            # 统一使用 TRANSCRIPTION_API_KEY 环境变量
            api_key = os.environ.get('TRANSCRIPTION_API_KEY', '')

            # 模型优先级：FUNASR_MODEL > TRANSCRIPTION_MODEL > 默认值
            model = 'fun-asr'
            funasr_model = os.environ.get('FUNASR_MODEL', '')
            if funasr_model:
                model = funasr_model
            else:
                transcription_model = os.environ.get('TRANSCRIPTION_MODEL', '')
                if transcription_model:
                    model = transcription_model

            # 解析语言提示数组（逗号分隔的字符串）
            language_hints_str = os.environ.get('ASR_LANGUAGE_HINTS', '')
            language_hints = []
            if language_hints_str:
                # 支持逗号分隔的字符串，如 "zh-CN,en-US,ja-JP"
                language_hints = [lang.strip() for lang in language_hints_str.split(',') if lang.strip()]

            return {
                'base_url': base_url,
                'api_key': api_key,
                'model': model,
                'timeout': self._get_asr_timeout(),
                'diarize': os.environ.get('ASR_DIARIZE', 'true').lower() == 'true',
                'disfluency_removal_enabled': os.environ.get('ASR_DISFLUENCY_REMOVAL_ENABLED', 'true').lower() == 'true',
                'timestamp_alignment_enabled': os.environ.get('ASR_TIMESTAMP_ALIGNMENT_ENABLED', 'false').lower() == 'true',
                'language_hints': language_hints
            }

        else:  # openai_whisper (default)
            base_url = os.environ.get('TRANSCRIPTION_BASE_URL', '')
            if base_url:
                base_url = base_url.split('#')[0].strip()

            # Support both TRANSCRIPTION_MODEL and legacy WHISPER_MODEL
            # TRANSCRIPTION_MODEL takes priority for custom Whisper variants
            model = os.environ.get('TRANSCRIPTION_MODEL', '') or os.environ.get('WHISPER_MODEL', 'whisper-1')

            return {
                'api_key': os.environ.get('TRANSCRIPTION_API_KEY', ''),
                'base_url': base_url or None,
                'model': model
            }

    def get_active_connector(self) -> BaseTranscriptionConnector:
        """
        Get the currently active connector.

        Initializes from environment if not already initialized.

        Returns:
            The active connector
        """
        if not self._active_connector:
            self.initialize_from_env()
        return self._active_connector

    def get_active_connector_name(self) -> str:
        """Get the name of the currently active connector."""
        if not self._active_connector:
            self.initialize_from_env()
        return self._connector_name

    def reinitialize(self) -> BaseTranscriptionConnector:
        """
        Force re-initialization of the connector.

        Useful when environment variables have changed.

        Returns:
            The newly initialized connector
        """
        self._active_connector = None
        self._connector_name = ""
        return self.initialize_from_env()


# Global registry instance
_registry: Optional[ConnectorRegistry] = None


def get_registry() -> ConnectorRegistry:
    """Get the global connector registry."""
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry()
    return _registry


# Convenience aliases
connector_registry = get_registry()


def transcribe(request: TranscriptionRequest) -> TranscriptionResponse:
    """
    Transcribe audio using the active connector.

    This is a convenience function that uses the global registry.

    Args:
        request: The transcription request

    Returns:
        Transcription response
    """
    connector = get_registry().get_active_connector()
    return connector.transcribe(request)


def get_connector() -> BaseTranscriptionConnector:
    """Get the active transcription connector."""
    return get_registry().get_active_connector()


def supports_diarization() -> bool:
    """Check if the active connector supports diarization."""
    return get_registry().get_active_connector().supports_diarization
