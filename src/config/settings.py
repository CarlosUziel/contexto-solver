import uuid  # Add import for uuid
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_host: str = Field(
        default="localhost",
        description="Hostname or IP address of the Qdrant server.",
    )
    # Renamed qdrant_port to reflect it's the host gRPC port for client connection
    qdrant_grpc_host_port: int = Field(
        default=6333,
        description="gRPC port exposed by the Qdrant host for client connections.",
    )
    qdrant_http_host_port: int = Field(
        default=6334,
        description="HTTP port exposed by the Qdrant host.",
    )
    # Added container ports for completeness, though not directly used by setup script
    qdrant_grpc_container_port: int = Field(
        default=6333,
        description="Internal gRPC port used by the Qdrant container.",
    )
    qdrant_http_container_port: int = Field(
        default=6334,
        description="Internal HTTP port used by the Qdrant container.",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="API key for authenticating with Qdrant Cloud (optional).",
    )
    qdrant_log_level: str = Field(
        default="info",
        description=(
            "Logging level for the Qdrant service "
            "(e.g., 'debug', 'info', 'warning', 'error')."
        ),
    )
    # Define a namespace for UUID generation (can be any valid UUID)
    qdrant_uuid_namespace: uuid.UUID = Field(
        default=uuid.NAMESPACE_DNS,
        description="Namespace for generating Qdrant point UUIDs.",
    )
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
    ] = Field(
        default="glove.6B.100d",
        description="Specifies the GloVe dataset to use for word embeddings.",
    )
    base_step_scale: float = Field(
        default=0.05, description="Base step scale for multiple guesses."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
