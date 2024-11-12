import json
from typing import Any, Dict, Type, TypeVar, Generic, Optional, Union
from dataclasses import asdict, fields
from enum import Enum
import os
from pathlib import Path
from datetime import datetime
from src.modules.data_format import DataFormat, OptionKey

T = TypeVar('T')


class DataConverter:
    """数据格式转换器，处理序列化和反序列化"""

    @staticmethod
    def _convert_enum(obj: Any) -> Any:
        """处理枚举类型的转换"""
        if isinstance(obj, Enum):
            return obj.value
        return obj

    @staticmethod
    def _convert_dataclass_to_dict(obj: Any) -> Dict:
        """将dataclass对象转换为字典"""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field in fields(obj):
                value = getattr(obj, field.name)
                result[field.name] = DataConverter.to_serializable(value)
            return result
        return obj

    @staticmethod
    def to_serializable(obj: Any) -> Any:
        """将对象转换为可序列化格式"""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, Enum):
            return DataConverter._convert_enum(obj)
        elif isinstance(obj, (list, tuple)):
            return [DataConverter.to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                DataConverter._convert_enum(k): DataConverter.to_serializable(v)
                for k, v in obj.items()
            }
        elif hasattr(obj, '__dataclass_fields__'):
            return DataConverter._convert_dataclass_to_dict(obj)
        else:
            raise TypeError(f"Unsupported type for serialization: {type(obj)}")

    @staticmethod
    def from_serializable(data: Any, target_type: Type[T]) -> T:
        """从可序列化格式转换回对象"""
        if data is None:
            return None

        if target_type == OptionKey:
            return OptionKey(data)

        if hasattr(target_type, '__dataclass_fields__'):
            field_types = {f.name: f.type for f in fields(target_type)}
            converted_data = {}

            for field_name, field_value in data.items():
                if field_name not in field_types:
                    continue

                field_type = field_types[field_name]

                # 处理Optional类型
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                    if type(None) in field_type.__args__:
                        if field_value is None:
                            converted_data[field_name] = None
                            continue
                        field_type = next(t for t in field_type.__args__ if t != type(None))

                # 处理List类型
                if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                    item_type = field_type.__args__[0]
                    converted_data[field_name] = [
                        DataConverter.from_serializable(item, item_type)
                        for item in field_value
                    ]
                # 处理Dict类型
                elif hasattr(field_type, '__origin__') and field_type.__origin__ is dict:
                    key_type, value_type = field_type.__args__
                    converted_data[field_name] = {
                        DataConverter.from_serializable(k, key_type):
                            DataConverter.from_serializable(v, value_type)
                        for k, v in field_value.items()
                    }
                # 处理tuple类型
                elif hasattr(field_type, '__origin__') and field_type.__origin__ is tuple:
                    converted_data[field_name] = tuple(
                        DataConverter.from_serializable(item, item_type)
                        for item, item_type in zip(field_value, field_type.__args__)
                    )
                # 处理基本类型和dataclass
                else:
                    converted_data[field_name] = (
                        DataConverter.from_serializable(field_value, field_type)
                        if hasattr(field_type, '__dataclass_fields__')
                        else field_value
                    )

            return target_type(**converted_data)

        return data


class CacheManager(Generic[T]):
    """通用缓存管理器"""

    def __init__(self, cache_dir: str, data_type: Type[T]):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
            data_type: 缓存数据类型
        """
        self.cache_dir = Path(cache_dir)
        self.data_type = data_type
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.json"

    def save(self, key: str, data: T) -> None:
        """
        保存数据到缓存

        Args:
            key: 缓存键
            data: 要缓存的数据
        """
        cache_path = self._get_cache_path(key)
        serialized_data = {
            'type': self.data_type.__name__,
            'timestamp': datetime.now().isoformat(),
            'data': DataConverter.to_serializable(data)
        }

        with cache_path.open('w', encoding='utf-8') as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=2)

    def load(self, key: str) -> Optional[T]:
        """
        从缓存加载数据

        Args:
            key: 缓存键

        Returns:
            Optional[T]: 缓存的数据，如果不存在则返回None
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with cache_path.open('r', encoding='utf-8') as f:
                cached = json.load(f)

            if cached['type'] != self.data_type.__name__:
                return None

            return DataConverter.from_serializable(cached['data'], self.data_type)
        except Exception as e:
            print(f"Error loading cache for {key}: {str(e)}")
            return None

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            bool: 是否存在缓存
        """
        return self._get_cache_path(key).exists()

    def clear(self, key: str = None) -> None:
        """
        清除缓存

        Args:
            key: 要清除的缓存键，如果为None则清除所有缓存
        """
        if key is None:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        else:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()


