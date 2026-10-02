import pytest

from crucible_entity_manager.crucible.sql import identifier, string_list, string_literal


@pytest.mark.parametrize("name", ["Entity_Stream_Manager_Configurations", "_x", "Feed2"])
def test_identifier_accepts_plain_names(name: str) -> None:
    assert identifier(name) == name


@pytest.mark.parametrize("name", ["", "2Feed", "Feed-A", "Feed A", "Feed;DROP", "Feed'"])
def test_identifier_rejects_anything_else(name: str) -> None:
    with pytest.raises(ValueError, match="valid dataset name"):
        identifier(name)


def test_string_literal_escapes_quotes() -> None:
    assert string_literal("O'Brien") == "'O''Brien'"


def test_string_list() -> None:
    assert string_list(["a", "b'c"]) == "('a', 'b''c')"
