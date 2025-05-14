import uuid  # Add import for uuid
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, configurable via environment variables or a .env file.

    These settings are automatically loaded by Pydantic from environment variables
    (matching the attribute names, case-insensitive) or from a `.env` file
    located in the project's root directory.

    Attributes:
        qdrant_host: Hostname or IP address of the Qdrant server.
        qdrant_grpc_host_port: gRPC port exposed by the Qdrant host for client connections.
        qdrant_http_host_port: HTTP port exposed by the Qdrant host.
        qdrant_grpc_container_port: Internal gRPC port used by the Qdrant container.
        qdrant_http_container_port: Internal HTTP port used by the Qdrant container.
        qdrant_api_key: API key for Qdrant Cloud (optional).
        qdrant_log_level: Logging level for the Qdrant service.
        qdrant_uuid_namespace: Namespace for generating Qdrant point UUIDs.
        glove_dataset: Specifies the GloVe dataset to use for word embeddings.
        qdrant_hnsw_ef: The 'ef' (size of the dynamic list for HNSW) parameter for Qdrant search.
        use_only_nouns: If true, the application will use a collection containing only nouns.
    """

    qdrant_host: str = Field(
        default="localhost",
        description="Hostname or IP address of the Qdrant server.",
    )
    qdrant_grpc_host_port: int = Field(
        default=6333,
        description="gRPC port exposed by the Qdrant host for client connections.",
    )
    qdrant_http_host_port: int = Field(
        default=6334,
        description="HTTP port exposed by the Qdrant host.",
    )
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
    qdrant_hnsw_ef: int = Field(
        default=64,
        description="The 'ef' (size of the dynamic list for HNSW) parameter for Qdrant search. Affects search speed and accuracy.",
    )
    use_only_nouns: bool = Field(
        default=False,
        description="If true, the application will use a Qdrant collection containing only nouns, suffixed with '_nouns'.",
    )

    @property
    def effective_collection_name(self) -> str:
        """str: Returns the actual Qdrant collection name to be used.

        The name is based on `glove_dataset` and `use_only_nouns` settings.
        If `use_only_nouns` is True, '_nouns' is appended to the dataset name.
        """
        if self.use_only_nouns:
            return f"{self.glove_dataset}_nouns"
        return self.glove_dataset

    class Config:
        """Pydantic model configuration.

        Attributes:
            env_file: The name of the environment file to load.
            env_file_encoding: The encoding of the environment file.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
