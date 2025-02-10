from typing import List
from learnware.learnware import Learnware


class BaseOrganizer:
    def __init__(self, type: str):
        self._type = type

    def add_learnware(self, learnware: Learnware) -> bool:
        raise NotImplementedError("add_learnware is not implemented")
    
    def get_learnware_list(self) -> List[Learnware]:
        raise NotImplementedError("get_learnware_list is not implemented")
    
    def get_learnware_count(self) -> int:
        raise NotImplementedError("get_learnware_count is not implemented")
    
    def get_learnware_ids(self) -> List[int]:
        raise NotImplementedError("get_learnware_ids is not implemented")

    @property
    def type(self) -> str:
        return self._type