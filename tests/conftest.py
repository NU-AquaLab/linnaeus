"""
Pytest configuration and shared fixtures.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from linnaeus.config.settings import Config
from linnaeus.models.schemas import (
    ASClassification,
    ASPOPData,
    ASRankData,
    ClassificationTags,
    OrganizationData,
    PeeringDBData,
)

#: Environment variables that Config._apply_env_overrides reads. A developer's
#: shell or the repo-root .env must never leak into test assertions (or test
#: failure output — .env may contain a real API key).
_CONFIG_ENV_VARS = [
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_BASE_URL",
    "OPENAI_ORG_ID",
    "DEFAULT_MODEL",
    "TEMPERATURE",
    "BATCH_SIZE",
    "MAX_CONCURRENT_FINE_TUNES",
    "DAILY_FINE_TUNE_LIMIT",
    "IPINFO_TOKEN",
    "DEFAULT_DATA_DIR",
    "CACHE_EXPIRY_HOURS",
    "LOG_LEVEL",
    "ENVIRONMENT",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Isolate every test from ambient config env vars and the root .env."""
    for var in _CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Config.from_file calls load_dotenv(), which would re-import the repo's
    # .env into os.environ mid-test; disable it entirely during tests.
    monkeypatch.setattr("linnaeus.config.settings.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration for testing."""
    config_data = {
        "environment": "testing",
        "openai": {
            "api_key": "test_key",
            "base_url": None,
            "default_model": "gpt-4o-mini",
            "temperature": 0.0001,
            "batch_size": 5,
        },
        "data": {
            "data_dir": str(temp_dir),
            "raw_data_dir": str(temp_dir / "raw"),
            "processed_data_dir": str(temp_dir / "processed"),
            "labeled_data_dir": str(temp_dir / "labeled"),
        },
        "apis": {
            "asrank_url": "https://test.api.com/graphql",
            "peeringdb_base_url": "https://test.peeringdb.com",
            "apnic_aspop_url": "https://test.apnic.com",
        },
    }

    # Create directories
    for dir_path in [temp_dir / "raw", temp_dir / "processed", temp_dir / "labeled"]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return config_data


@pytest.fixture
def mock_config(sample_config):
    """Mock configuration for tests."""
    with patch("linnaeus.config.settings.get_config") as mock_get_config:
        config = Config(**sample_config)
        mock_get_config.return_value = config
        yield config


@pytest.fixture
def sample_asrank_data():
    """Create sample ASRank data."""
    return ASRankData(
        asn=174,
        asn_name="COGENT-174",
        rank=10,
        organization_name="Cogent Communications",
        clique_member=True,
        seen=True,
        longitude=-77.0365,
        latitude=38.8951,
        country_iso="US",
        country_name="United States",
        cone_asns=15000,
        cone_prefixes=50000,
        cone_addresses=1000000,
        degree_provider=5,
        degree_peer=100,
        degree_customer=500,
        degree_total=605,
        announcing_prefixes=1000,
        announcing_addresses=50000,
    )


@pytest.fixture
def sample_peeringdb_data():
    """Create sample PeeringDB data."""
    return PeeringDBData(
        asn=174,
        name="Cogent Communications",
        organization="Cogent Communications, Inc.",
        website="https://www.cogentco.com",
        looking_glass="https://lg.cogentco.com",
        info_type="NSP",
        info_scope="Global",
        info_unicast=True,
        info_multicast=False,
        info_ipv6=True,
        policy_general="Open",
    )


@pytest.fixture
def sample_aspop_data():
    """Create sample ASPOP data."""
    return ASPOPData(
        asn=174,
        name="COGENT-174",
        country="US",
        rir="ARIN",
        customer_cone_asns=12000,
        customer_cone_prefixes=45000,
        customer_cone_addresses=900000,
    )


@pytest.fixture
def sample_organization_data(
    sample_asrank_data, sample_peeringdb_data, sample_aspop_data
):
    """Create sample organization data."""
    return OrganizationData(
        asn=174,
        asrank=sample_asrank_data,
        peeringdb=sample_peeringdb_data,
        aspop=sample_aspop_data,
    )


@pytest.fixture
def sample_classification():
    """Create sample classification result."""
    return ASClassification(
        asn=174,
        organization_name="Cogent Communications",
        tags=[ClassificationTags.TRANSIT, ClassificationTags.ENTERPRISE],
        confidence_scores={"Transit": 0.95, "Enterprise": 0.75},
        model_used="gpt-4o-mini-test",
    )


@pytest.fixture
def sample_labeled_data():
    """Create sample labeled training data."""
    data = {
        "asn": [174, 15169, 32934],
        "organization_name": ["Cogent Communications", "Google LLC", "Facebook, Inc."],
        "country": ["US", "US", "US"],
        "website": [
            "https://www.cogentco.com",
            "https://www.google.com",
            "https://www.facebook.com",
        ],
        "Transit": [1, 0, 0],
        "ContentProvider": [0, 1, 1],
        "Enterprise": [1, 1, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()

    # Mock completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.parsed = Mock()
    mock_response.choices[0].message.parsed.responses = [
        Mock(tags=[ClassificationTags.TRANSIT])
    ]
    mock_response.choices[0].logprobs = Mock()
    mock_response.choices[0].logprobs.content = []

    mock_client.beta.chat.completions.parse.return_value = mock_response

    # Mock file operations
    mock_file = Mock()
    mock_file.id = "file-test123"
    mock_client.files.create.return_value = mock_file

    # Mock fine-tuning
    mock_job = Mock()
    mock_job.id = "ftjob-test123"
    mock_job.status = "succeeded"
    mock_job.fine_tuned_model = "ft:gpt-4o-mini-test"
    mock_job.trained_tokens = 1000
    mock_client.fine_tuning.jobs.create.return_value = mock_job
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    return mock_client


@pytest.fixture
def sample_csv_file(temp_dir, sample_labeled_data):
    """Create a sample CSV file for testing."""
    csv_path = temp_dir / "test_data.csv"
    sample_labeled_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for API responses."""
    return {
        "data": {
            "asns": {
                "totalCount": 2,
                "pageInfo": {"first": 2, "hasNextPage": False},
                "edges": [
                    {
                        "node": {
                            "asn": 174,
                            "asnName": "COGENT-174",
                            "rank": 10,
                            "organization": {
                                "orgId": "COGENT",
                                "orgName": "Cogent Communications",
                            },
                            "cliqueMember": True,
                            "seen": True,
                            "longitude": -77.0365,
                            "latitude": 38.8951,
                            "country": {"iso": "US", "name": "United States"},
                            "cone": {
                                "numberAsns": 15000,
                                "numberPrefixes": 50000,
                                "numberAddresses": 1000000,
                            },
                            "asnDegree": {
                                "provider": 5,
                                "peer": 100,
                                "customer": 500,
                                "total": 605,
                                "transit": 505,
                                "sibling": 0,
                            },
                            "announcing": {
                                "numberPrefixes": 1000,
                                "numberAddresses": 50000,
                            },
                        }
                    },
                    {
                        "node": {
                            "asn": 15169,
                            "asnName": "GOOGLE",
                            "rank": 1,
                            "organization": {
                                "orgId": "GOOGLE",
                                "orgName": "Google LLC",
                            },
                            "cliqueMember": False,
                            "seen": True,
                            "longitude": -122.0838,
                            "latitude": 37.4220,
                            "country": {"iso": "US", "name": "United States"},
                            "cone": {
                                "numberAsns": 500,
                                "numberPrefixes": 10000,
                                "numberAddresses": 500000,
                            },
                            "asnDegree": {
                                "provider": 10,
                                "peer": 200,
                                "customer": 100,
                                "total": 310,
                                "transit": 110,
                                "sibling": 0,
                            },
                            "announcing": {
                                "numberPrefixes": 5000,
                                "numberAddresses": 250000,
                            },
                        }
                    },
                ],
            }
        }
    }


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for async HTTP requests."""
    mock_session = Mock()
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json.return_value = {"test": "data"}
    mock_response.text.return_value = "test,data\n1,2\n3,4"

    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.head.return_value.__aenter__.return_value = mock_response

    return mock_session


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global configuration before each test."""
    # Clear the global config cache
    import linnaeus.config.settings

    linnaeus.config.settings._config = None
    yield
    linnaeus.config.settings._config = None
