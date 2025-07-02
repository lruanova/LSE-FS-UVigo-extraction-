from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.transformations.operators.base_operator import (
    BaseOperator,
)


class Pipeline:
    def __init__(self, transformations: list[BaseOperator]):
        self.transformations = transformations
        self.validate()

    def __call__(self, annotation: AnnotationRecord):
        for t_operator_instance in self.transformations:
            t_operator_instance.apply(annotation)
        return annotation

    def validate(self):
        """Validates that the dependencies of each operator are met by preceding operators in the pipeline."""
        processed_operator_types = set()
        for idx, operator in enumerate(self.transformations):
            current_op_dependencies = getattr(operator, "dependencies", [])

            for dependency_type in current_op_dependencies:
                if dependency_type not in processed_operator_types:
                    raise RuntimeError(
                        f"Operator '{operator.__class__.__name__}' (at index {idx}) requires "
                        f"'{dependency_type.__name__}' to precede it, but it was not found or "
                        f"not processed yet in the pipeline."
                    )

            processed_operator_types.add(type(operator))
