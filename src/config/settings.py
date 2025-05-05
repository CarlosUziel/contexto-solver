from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    # Renamed qdrant_port to reflect it's the host gRPC port for client connection
    qdrant_grpc_host_port: int = 6333
    qdrant_http_host_port: int = 6334
    # Added container ports for completeness, though not directly used by setup script
    qdrant_grpc_container_port: int = 6333
    qdrant_http_container_port: int = 6334
    qdrant_api_key: str | None = None
    qdrant_log_level: str = "info"  # Add this line
    glove_dataset: Literal[
        "glove.6B.50d",
        "glove.6B.100d",
        "glove.6B.200d",
        "glove.6B.300d",
        "glove.twitter.27B.25d",
        "glove.twitter.27B.50d",
        "glove.twitter.27B.100d",
        "glove.twitter.27B.200d",
        "glove.42B.300d",
        "glove.840B.300d",
    ] = "glove.6B.100d"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
