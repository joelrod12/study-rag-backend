from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """
    Domain entity representing an indexed document.
    """

    id: str

    filename: str

    file_path: str

    created_at: datetime

    is_active: bool = True
