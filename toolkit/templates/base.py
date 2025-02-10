import os
import tempfile
from shutil import copyfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from learnware.utils import convert_folder_to_zipfile
from learnware.tests.templates import ModelTemplate, LearnwareTemplate, StatSpecTemplate, PickleModelTemplate


@dataclass
class OurModelTemplate(ModelTemplate):
    class_name: str # [JoblibLoadedModel]
    model_filepath: str
    model_kwargs: dict
    template_path: str = field(init=False)

    def __post_init__(self):
        template_path = {
            "JoblibLoadedModel": "joblib_model.py",
        }
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.template_path = os.path.join(dir_path, template_path[self.class_name])
        default_model_kwargs = {
            "predict_method": "predict",
            "fit_method": "fit",
            "finetune_method": "finetune",
            "model_filename": "model.out",
        }
        default_model_kwargs.update(self.model_kwargs)
        self.model_kwargs = default_model_kwargs

        
class OurLearnwareTemplate(LearnwareTemplate):
    @staticmethod
    def generate_learnware_zipfile(
        learnware_zippath: str,
        model_template: ModelTemplate,
        stat_spec_templates: List[StatSpecTemplate],
        requirements: Optional[List[Union[Tuple[str, str, str], str]]] = None,
    ):
        with tempfile.TemporaryDirectory(suffix="learnware_template") as tempdir:
            requirement_filepath = os.path.join(tempdir, "requirements.txt")
            LearnwareTemplate.generate_requirements(requirement_filepath, requirements)

            model_filepath = os.path.join(tempdir, "__init__.py")
            copyfile(model_template.template_path, model_filepath)

            learnware_yaml_filepath = os.path.join(tempdir, "learnware.yaml")
            model_config = {
                "class_name": model_template.class_name,
                "kwargs": model_template.model_kwargs,
            }

            stat_spec_config_list = []
            for stat_spec_template in stat_spec_templates:
                stat_spec_config = {
                    "module_path": "learnware.specification",
                    "class_name": stat_spec_template.type,
                    "file_name": os.path.basename(stat_spec_template.filepath),
                    "kwargs": {},
                }
                copyfile(stat_spec_template.filepath, os.path.join(tempdir, stat_spec_config["file_name"]))
                stat_spec_config_list.append(stat_spec_config)
            LearnwareTemplate.generate_learnware_yaml(learnware_yaml_filepath, model_config, stat_spec_config=stat_spec_config_list)

            if isinstance(model_template, OurModelTemplate):
                model_filepath = os.path.join(tempdir, model_template.model_kwargs["model_filename"])
                copyfile(model_template.model_filepath, model_filepath)

            convert_folder_to_zipfile(tempdir, learnware_zippath)
