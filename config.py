# config.py - Secure API key configuration
"""
Secure API key configuration using environment variables.
Never commit this file to version control!
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for API keys and settings"""

    # LLM Provider API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

    # Third-party Provider API Keys
    THIRD_PARTY_API_KEY = os.getenv('THIRD_PARTY_API_KEY')
    THIRD_PARTY_API_BASE_URL = os.getenv('THIRD_PARTY_API_BASE_URL')

    # Model Settings
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 1000))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))

    @classmethod
    def get_api_key(cls, provider='third_party'):
        """Get API key for specified provider"""
        provider_map = {
            'openai': cls.OPENAI_API_KEY,
            'anthropic': cls.ANTHROPIC_API_KEY,
            'huggingface': cls.HUGGINGFACE_TOKEN,
            'third_party': cls.THIRD_PARTY_API_KEY
        }

        key = provider_map.get(provider.lower())
        if not key or key == 'your_api_key_here':
            raise ValueError(f"API key for {provider} not found or not configured")

        return key
    
    @classmethod
    def get_temperature(cls, privider='third_party'):
        """Get temperature setting for specified provider"""
        return cls.TEMPERATURE
        

    @classmethod
    def validate_keys(cls, required_providers=None):
        """Validate that required API keys are present"""
        if required_providers is None:
            required_providers = ['third_party']  # Default to your current provider

        missing_keys = []

        for provider in required_providers:
            try:
                cls.get_api_key(provider)
            except ValueError:
                missing_keys.append(provider)

        if missing_keys:
            raise ValueError(f"Missing API keys for providers: {', '.join(missing_keys)}")

        return True

    @classmethod
    def get_headers(cls, provider='third_party'):
        """Get HTTP headers for API requests"""
        api_key = cls.get_api_key(provider)

        headers = {
            'Content-Type': 'application/json'
        }

        if provider.lower() == 'openai':
            headers['Authorization'] = f'Bearer {api_key}'
        elif provider.lower() == 'anthropic':
            headers['x-api-key'] = api_key
            headers['anthropic-version'] = '2023-06-01'
        elif provider.lower() == 'huggingface':
            headers['Authorization'] = f'Bearer {api_key}'
        elif provider.lower() == 'third_party':
            # Adjust this based on your third-party provider's requirements
            headers['Authorization'] = f'Bearer {api_key}'
            headers['X-API-Key'] = api_key

        return headers

    @classmethod
    def get_api_url(cls, provider='third_party', endpoint=None):
        """Get API URL for requests"""
        base_urls = {
            'openai': 'https://api.openai.com/v1',
            'anthropic': 'https://api.anthropic.com',
            'huggingface': 'https://api-inference.huggingface.co',
            'third_party': cls.THIRD_PARTY_API_BASE_URL or 'https://api.siliconflow.cn/v1'
        }

        base_url = base_urls.get(provider.lower())
        if not base_url:
            raise ValueError(f"Unknown provider: {provider}")

        # For Silicon Flow API, the base URL should be used directly without endpoint
        if provider.lower() == 'third_party' and endpoint:
            # Silicon Flow might use different endpoint format
            # Try without endpoint first, or with a different format
            return f"{base_url}/{endpoint}"
        elif endpoint:
            return f"{base_url}/{endpoint}"
        else:
            return base_url
    
    @classmethod
    def get_default_model(cls):
        """Get the default model name"""
        return cls.DEFAULT_MODEL

# Example usage
if __name__ == "__main__":
    try:
        Config.validate_keys(['third_party'])
        print("✅ API keys validated successfully")

        # Example: Get headers for your current provider
        headers = Config.get_headers('third_party')
        print(f"Headers for third-party API: {headers}")

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease check your .env file and ensure all required API keys are set.")
        print("Use .env.example as a template.")