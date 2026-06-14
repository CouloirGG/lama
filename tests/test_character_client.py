"""Tests for character_client.py — GGG Character API parsing."""

import pytest
from unittest.mock import MagicMock

from character_client import CharacterClient


@pytest.fixture
def client():
    """Create a CharacterClient with a mock OAuthManager."""
    mock_oauth = MagicMock()
    mock_oauth.connected = True
    mock_oauth.get_headers.return_value = {
        "Authorization": "Bearer fake-token",
        "User-Agent": "test",
    }
    return CharacterClient(mock_oauth)


# -------------------------------------------------------------------
# _parse_equipment_item tests
# -------------------------------------------------------------------

class TestParseEquipmentItem:
    def test_rare_body_armour(self):
        item = {
            "name": "Pandemonium Suit",
            "typeLine": "Expert Hexer's Robe",
            "inventoryId": "BodyArmour",
            "frameType": 2,
            "implicitMods": ["+20 to Intelligence"],
            "explicitMods": [
                "+84 to maximum Life",
                "+42% to Fire Resistance",
                "+33% to Cold Resistance",
            ],
        }
        eq = CharacterClient._parse_equipment_item(item)
        assert eq is not None
        assert eq.name == "Pandemonium Suit"
        assert eq.type_line == "Expert Hexer's Robe"
        assert eq.slot == "BodyArmour"
        assert eq.rarity == "rare"
        assert len(eq.implicit_mods) == 1
        assert len(eq.explicit_mods) == 3

    def test_unique_weapon(self):
        item = {
            "name": "The Annihilator",
            "typeLine": "Greataxe",
            "inventoryId": "Weapon",
            "frameType": 3,
            "implicitMods": [],
            "explicitMods": [
                "Adds 50 to 100 Physical Damage",
                "20% increased Attack Speed",
            ],
            "sockets": [{"group": 0}, {"group": 0}],
        }
        eq = CharacterClient._parse_equipment_item(item)
        assert eq is not None
        assert eq.rarity == "unique"
        assert eq.slot == "Weapon"
        assert len(eq.sockets) == 2

    def test_normal_item(self):
        item = {
            "name": "",
            "typeLine": "Iron Ring",
            "inventoryId": "Ring",
            "frameType": 0,
        }
        eq = CharacterClient._parse_equipment_item(item)
        assert eq is not None
        assert eq.rarity == "normal"
        assert eq.slot == "Ring"

    def test_empty_item(self):
        eq = CharacterClient._parse_equipment_item({})
        assert eq is None

    def test_none_item(self):
        eq = CharacterClient._parse_equipment_item(None)
        assert eq is None

    def test_no_inventory_id(self):
        item = {"name": "Test", "typeLine": "Test", "frameType": 2}
        eq = CharacterClient._parse_equipment_item(item)
        assert eq is None

    def test_crafted_and_fractured_mods(self):
        item = {
            "name": "Storm Brow",
            "typeLine": "Great Helmet",
            "inventoryId": "Helm",
            "frameType": 2,
            "craftedMods": ["+25 to maximum Life"],
            "fracturedMods": ["+15% to Lightning Resistance"],
            "explicitMods": ["+30 to Strength"],
        }
        eq = CharacterClient._parse_equipment_item(item)
        assert eq is not None
        assert len(eq.crafted_mods) == 1
        assert len(eq.fractured_mods) == 1
        assert len(eq.explicit_mods) == 1


# -------------------------------------------------------------------
# _parse_character tests
# -------------------------------------------------------------------

class TestParseCharacter:
    def test_basic_character(self, client):
        data = {
            "name": "TestWitch",
            "class": "Blood Mage",
            "level": 85,
            "equipment": [
                {
                    "name": "Pandemonium Suit",
                    "typeLine": "Expert Hexer's Robe",
                    "inventoryId": "BodyArmour",
                    "frameType": 2,
                    "explicitMods": ["+84 to maximum Life"],
                },
                {
                    "name": "Storm Grip",
                    "typeLine": "Leather Gloves",
                    "inventoryId": "Gloves",
                    "frameType": 2,
                    "explicitMods": ["+30 to Dexterity"],
                },
            ],
            "skills": [
                {
                    "gems": [
                        {"name": "Arc"},
                        {"name": "Added Lightning Damage Support"},
                    ]
                }
            ],
        }
        char = client._parse_character(data)
        assert char is not None
        assert char.name == "TestWitch"
        assert char.char_class == "Witch"  # Blood Mage → Witch
        assert char.ascendancy == "Blood Mage"
        assert char.level == 85
        assert len(char.equipment) == 2
        assert len(char.skill_groups) == 1
        assert "Arc" in char.skill_groups[0].gems
        assert len(char.skill_groups[0].dps) == 0  # GGG API has no DPS
        assert char.keystones == []

    def test_non_ascendancy_class(self, client):
        data = {
            "name": "RawWitch",
            "class": "Witch",
            "level": 10,
            "equipment": [],
        }
        char = client._parse_character(data)
        assert char is not None
        assert char.char_class == "Witch"
        assert char.ascendancy == "Witch"

    def test_empty_equipment(self, client):
        data = {
            "name": "Naked",
            "class": "Warrior",
            "level": 1,
        }
        char = client._parse_character(data)
        assert char is not None
        assert len(char.equipment) == 0
        assert len(char.skill_groups) == 0

    def test_account_as_dict(self, client):
        data = {
            "name": "TestChar",
            "class": "Titan",
            "level": 90,
            "account": {"name": "MyAccount"},
            "equipment": [],
        }
        char = client._parse_character(data)
        assert char is not None
        assert char.account == "MyAccount"
        assert char.char_class == "Warrior"  # Titan → Warrior

    def test_account_as_string(self, client):
        data = {
            "name": "TestChar",
            "class": "Deadeye",
            "level": 80,
            "account": "StringAccount",
            "equipment": [],
        }
        char = client._parse_character(data)
        assert char is not None
        assert char.account == "StringAccount"
        assert char.char_class == "Ranger"  # Deadeye → Ranger

    def test_items_key_fallback(self, client):
        """GGG API may use 'items' instead of 'equipment'."""
        data = {
            "name": "AltFormat",
            "class": "Sorceress",
            "level": 70,
            "items": [
                {
                    "name": "Test Ring",
                    "typeLine": "Gold Ring",
                    "inventoryId": "Ring",
                    "frameType": 2,
                    "explicitMods": ["+30 to all Resistances"],
                },
            ],
        }
        char = client._parse_character(data)
        assert char is not None
        assert len(char.equipment) == 1
        assert char.equipment[0].slot == "Ring"

    def test_skills_with_base_type_fallback(self, client):
        """Gems may use baseType instead of name."""
        data = {
            "name": "GemTest",
            "class": "Monk",
            "level": 60,
            "equipment": [],
            "skills": [
                {
                    "gems": [
                        {"baseType": "Fireball"},
                        {"name": "Spell Echo Support"},
                    ]
                }
            ],
        }
        char = client._parse_character(data)
        assert char is not None
        assert len(char.skill_groups) == 1
        assert "Fireball" in char.skill_groups[0].gems
        assert "Spell Echo Support" in char.skill_groups[0].gems
