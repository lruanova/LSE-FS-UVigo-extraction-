from abc import ABC, abstractmethod

from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


class BaseOperator(ABC):
    def __init__(self, subsets_to_apply: list[str]):
        super().__init__()
        self.subsets_to_apply = subsets_to_apply

    @abstractmethod
    def apply(self, ann: AnnotationRecord) -> AnnotationRecord:
        pass

    @property
    def dependencies(self) -> list[type]:
        return []

    @staticmethod
    def is_valid(p):
        return p.x is not None and p.y is not None and p.z is not None

    def __call__(self, ann: AnnotationRecord):
        self.apply(ann)
        return ann

    def __repr__(self):
        return f"{self.__class__.__name__}(subsets_to_apply={self.subsets_to_apply})"
