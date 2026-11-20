import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import app


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Next_best_article_recommendation (3)_article_level.ipynb"


def notebook_cell(marker):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if marker in "".join(cell.get("source", []))
    )


@unittest.skipUnless(NOTEBOOK.is_file(), "Notebook is not present in this checkout")
class ArtifactMappingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        directory = Path(self.directory.name)
        namespace = {}
        exec(notebook_cell("section_mapping = {"), namespace)
        self.mapping = namespace["section_mapping"]
        self.assertEqual(self.mapping["Life &amp; Style"], "Leisure")
        self.assertEqual(self.mapping["Life & Style"], "Leisure")

        model = SimpleNamespace(save=lambda path: Path(path).touch())
        joblib = SimpleNamespace(dump=lambda obj, path: Path(path).touch())
        namespace.update(
            {
                "model": model,
                "joblib": joblib,
                "transformer": object(),
                "user_customer_article_interactions_clean": pd.DataFrame(
                    {
                        "articleid": ["1", "1", "2", "3", "4", None],
                        "page_title": [None, "Shared title", "Shared title", "Other title", " ", "Ignored"],
                    }
                ),
                "pd": pd,
                "json": json,
                "ARTICLE_TO_INDEX": {"1": 0, "2": 1},
                "SECTION_TO_INDEX": {"Leisure": 0},
                "LOOK_BACK": 5,
                "profile_cols": ["profile", *app.TIME_FEATURES],
                "tf": SimpleNamespace(__version__="test"),
                "sklearn": SimpleNamespace(__version__="test"),
                "np": np,
            }
        )
        previous = Path.cwd()
        try:
            os.chdir(directory)
            exec(notebook_cell('MODEL_PATH = "article_recommender_article_level.keras"\nPREPROCESSOR_PATH'), namespace)
        finally:
            os.chdir(previous)
        self.metadata_path = directory / "article_recommender_article_level_metadata.json"
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def load_resources(self):
        preprocessor = SimpleNamespace(get_feature_names_out=lambda: ["profile"])
        model = SimpleNamespace(num_articles=2, num_sections=1, profile_dim=5)
        with (
            patch.dict(os.environ, {"MODEL_ARTIFACTS_DIR": self.directory.name}),
            patch.object(app.joblib, "load", return_value=preprocessor),
            patch.object(app.tf.keras.models, "load_model", return_value=model),
        ):
            return app.load_resources()

    def test_export_and_search_duplicate_titles(self):
        self.assertEqual(self.metadata["section_mapping"], self.mapping)
        self.assertEqual(
            self.metadata["title_to_article_ids"],
            {"Shared title": ["1", "2"], "Other title": ["3"]},
        )
        resources = self.load_resources()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(resources=resources)))
        results = app.search_articles(request, q="SHARED", limit=8)
        self.assertEqual([(result.article_id, result.title) for result in results], [
            ("1", "Shared title"),
            ("2", "Shared title"),
        ])
        self.assertEqual(app.search_articles(request, q="Other", limit=8), [])

    def test_older_metadata_derives_title_index(self):
        del self.metadata["title_to_article_ids"]
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        self.assertEqual(self.load_resources().title_to_article_ids["Shared title"], ["1", "2"])

    def test_inconsistent_title_index_is_rejected(self):
        self.metadata["title_to_article_ids"]["Shared title"] = ["1"]
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "title-to-article mapping"):
            self.load_resources()


if __name__ == "__main__":
    unittest.main()
