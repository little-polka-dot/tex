from importlib import import_module
from typing import Dict, Any
from copy import deepcopy


def import_class(module_cls: str):
    if module_cls[0] == '.':
        module_cls = '__main__' + module_cls
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
        settings: Dict[str, Any], cls='class'):
    """
    递归扫描dict，动态导入其中的cls并构建对象。
    >> settings = {"YourObject": {"class": "YourModule.YourClass", ...}}
    >> build_from_settings(settings)
    {"YourObject": YourModule.YourClass(...)}
    class字段格式: [module.]module.class[.class]
    当类实现在执行文件中时可缺省模块名: .class[.class]
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
    print(import_class('.A.B'))
