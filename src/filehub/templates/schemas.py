"""Template schemas for project generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DirectoryNode:
    """Directory structure node."""

    name: str
    children: list[DirectoryNode] = field(default_factory=list)
    files: list[FileNode] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirectoryNode:
        """Create DirectoryNode from dictionary."""
        name = data.get("name", "")
        children = [cls.from_dict(c) for c in data.get("children", [])]
        files = [FileNode.from_dict(f) for f in data.get("files", [])]
        return cls(name=name, children=children, files=files)


@dataclass
class FileNode:
    """File node template."""

    name: str
    content: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileNode:
        """Create FileNode from dictionary."""
        return cls(name=data.get("name", ""), content=data.get("content", ""))


@dataclass
class ProjectTemplate:
    """Project template definition."""

    name: str
    description: str
    structure: DirectoryNode

    naming_profile: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectTemplate:
        """Create ProjectTemplate from dictionary."""
        return cls(
            name=data.get("name", "standard"),
            description=data.get("description", ""),
            structure=DirectoryNode.from_dict(data.get("structure", {})),
            naming_profile=data.get("naming_profile"),
        )


@dataclass
class OrganizeTemplate:
    """File organization template with classification rules.

    Attributes:
        name: Template identifier
        description: Human-readable description
        ext_groups: Extension-to-group mapping (e.g. {"drawings": ["dwg", "dxf"]})
        rules: ActionRule dicts for file organization
        default_group: Fallback group for unmapped extensions
        folder_template: Optional ProjectTemplate name for pre-scaffolding
    """

    name: str
    description: str = ""
    ext_groups: dict[str, list[str]] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)
    default_group: str = "others"
    folder_template: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrganizeTemplate:
        """Create OrganizeTemplate from dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            ext_groups=data.get("ext_groups", {}),
            rules=data.get("rules", []),
            default_group=data.get("default_group", "others"),
            folder_template=data.get("folder_template"),
        )
