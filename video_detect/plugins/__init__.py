"""Plugin system for extensible detectors and processors"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Type
import importlib.util
import inspect
import logging

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugins"""

    def __init__(self):
        self.detectors: Dict[str, Type] = {}
        self.processors: Dict[str, Type] = {}
        self.transformers: Dict[str, Type] = {}
        self.analyzers: Dict[str, Type] = {}

    def register_detector(
        self,
        name: str,
        detector_class: Type,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a detector plugin"""
        self.detectors[name] = detector_class
        logger.info(f"Registered detector plugin: {name}")

    def register_processor(
        self,
        name: str,
        processor_class: Type,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a processor plugin"""
        self.processors[name] = processor_class
        logger.info(f"Registered processor plugin: {name}")

    def register_transformer(
        self,
        name: str,
        transformer_class: Type,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a transformer plugin"""
        self.transformers[name] = transformer_class
        logger.info(f"Registered transformer plugin: {name}")

    def register_analyzer(
        self,
        name: str,
        analyzer_class: Type,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an analyzer plugin"""
        self.analyzers[name] = analyzer_class
        logger.info(f"Registered analyzer plugin: {name}")

    def get_detector(self, name: str) -> Optional[Type]:
        """Get detector class by name"""
        return self.detectors.get(name)

    def get_processor(self, name: str) -> Optional[Type]:
        """Get processor class by name"""
        return self.processors.get(name)

    def get_transformer(self, name: str) -> Optional[Type]:
        """Get transformer class by name"""
        return self.transformers.get(name)

    def get_analyzer(self, name: str) -> Optional[Type]:
        """Get analyzer class by name"""
        return self.analyzers.get(name)

    def list_detectors(self) -> List[str]:
        """List all registered detectors"""
        return list(self.detectors.keys())

    def list_processors(self) -> List[str]:
        """List all registered processors"""
        return list(self.processors.keys())

    def list_transformers(self) -> List[str]:
        """List all registered transformers"""
        return list(self.transformers.keys())

    def list_analyzers(self) -> List[str]:
        """List all registered analyzers"""
        return list(self.analyzers.keys())


class PluginLoader:
    """Load plugins from Python files"""

    def __init__(self, registry: PluginRegistry):
        self.registry = registry

    def load_from_directory(self, plugin_dir: Path) -> Dict[str, Any]:
        """Load all plugins from a directory"""
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return {}

        loaded_plugins = {}

        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                plugin_info = self.load_plugin_file(plugin_file)
                if plugin_info:
                    loaded_plugins[plugin_file.stem] = plugin_info
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file.name}: {e}")

        return loaded_plugins

    def load_plugin_file(self, plugin_file: Path) -> Optional[Dict[str, Any]]:
        """Load a single plugin file"""
        module_name = f"plugin_{plugin_file.stem}"

        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find and register plugin classes
        plugin_info = {
            "file": str(plugin_file),
            "detectors": [],
            "processors": [],
            "transformers": [],
            "analyzers": [],
        }

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, "__plugin_metadata__"):
                metadata = obj.__plugin_metadata__
                plugin_type = metadata.get("type")

                if plugin_type == "detector":
                    self.registry.register_detector(name, obj, metadata)
                    plugin_info["detectors"].append(name)
                elif plugin_type == "processor":
                    self.registry.register_processor(name, obj, metadata)
                    plugin_info["processors"].append(name)
                elif plugin_type == "transformer":
                    self.registry.register_transformer(name, obj, metadata)
                    plugin_info["transformers"].append(name)
                elif plugin_type == "analyzer":
                    self.registry.register_analyzer(name, obj, metadata)
                    plugin_info["analyzers"].append(name)

        return plugin_info if any(plugin_info.values()[1:]) else None


# Global plugin registry
_plugin_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get global plugin registry"""
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry


def setup_plugins(plugin_dir: Optional[Path] = None) -> PluginRegistry:
    """Setup and load plugins"""
    registry = get_plugin_registry()

    if plugin_dir:
        loader = PluginLoader(registry)
        loader.load_from_directory(plugin_dir)

    return registry


# Plugin base classes for users to extend
class DetectorPlugin:
    """Base class for detector plugins"""

    __plugin_metadata__ = {"type": "detector"}

    def detect(self, file_path: Path) -> Dict[str, Any]:
        """Detect watermarks/fingerprints"""
        raise NotImplementedError

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return configuration schema for the plugin"""
        return {}


class ProcessorPlugin:
    """Base class for processor plugins"""

    __plugin_metadata__ = {"type": "processor"}

    def process(self, input_path: Path, output_path: Path) -> bool:
        """Process file to remove watermarks/fingerprints"""
        raise NotImplementedError

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return configuration schema for the plugin"""
        return {}


class TransformerPlugin:
    """Base class for transformer plugins"""

    __plugin_metadata__ = {"type": "transformer"}

    def transform(self, input_path: Path, output_path: Path) -> bool:
        """Transform file to change signature"""
        raise NotImplementedError

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return configuration schema for the plugin"""
        return {}


class AnalyzerPlugin:
    """Base class for analyzer plugins"""

    __plugin_metadata__ = {"type": "analyzer"}

    def analyze(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file for patterns"""
        raise NotImplementedError

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return configuration schema for the plugin"""
        return {}
