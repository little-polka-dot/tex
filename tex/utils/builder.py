from importlib import import_module
from typing import Dict, Any
from copy import deepcopy


def import_class(module_cls):
    if module_cls[0] == '.':
        # 局部路径则采用当前项目为根路径
        module_cls = 'tex' + module_cls
    module_cls = module_cls.split('.')
    module_idx = len(module_cls) - 1
    module = None
    while module_idx > 0:
        # 尝试区分module与class并加载module
        # 首先默认末尾为class 其余为module
        try:
            module = import_module(
                '.'.join(module_cls[:module_idx]))
        except ModuleNotFoundError:
            module_idx -= 1
        else: break
    if module:
        for cls in module_cls[module_idx:]:
            module = getattr(module, cls)
        return module


def build_from_settings(
        settings: Dict[str, Any], cls='cls'):
    """
    递归扫描dict 如果存在cls字段 则动态导入该类并初始化对象
    """
    def _build(root):
        for key in root:
            if isinstance(root[key], dict):
                root[key] = _build(root[key])
        if cls in root:
            return import_class(
                root.pop(cls))(**root)
        else:
            return root
    if isinstance(settings, dict):
        return _build(deepcopy(settings))


if __name__ == '__main__':
    print(import_class('A.B'))
