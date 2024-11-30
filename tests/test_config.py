import pytest
from config import config


def test_database_config():
    # Test default database configuration
    db_config = config.get_db_config()
    assert db_config["database"] == "casual"

    # Test switching database
    db_config = config.get_db_config("casual_1")
    assert db_config["database"] == "casual-1"

    # Test invalid database
    with pytest.raises(ValueError):
        config.get_db_config("invalid_db")


def test_paths():
    paths = config.get_paths()
    assert "data" in paths
    assert "cache" in paths
    assert "output" in paths
    assert "logs" in paths

    # Test directory creation
    for path in paths.values():
        assert path.exists()


def test_logger():
    logger = config.get_logger("20-gpt-4-adaptive-knowledge-0.75-shortest-enhance-ultra2")
    assert logger.name == "casual_graphrag.20-gpt-4-adaptive-knowledge-0.75-shortest-enhance-ultra2"

    # Test log file creation
    log_path = config.paths["logs"] / "medical_qa.log"
    assert log_path.exists()