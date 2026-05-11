"""Tests for sim/elements.py."""

from __future__ import annotations
import pytest
from sim.elements import (
    ELEMENTS, ELEMENTS_LIST, SYMBOL_TO_ID,
    fusion_product, get,
)


class TestElementTable:
    def test_key_elements_present(self):
        for sym in ('H', 'He', 'C', 'O', 'Fe', 'D'):
            assert sym in ELEMENTS

    def test_elements_list_matches_dict(self):
        assert len(ELEMENTS_LIST) == len(ELEMENTS)
        for elem in ELEMENTS_LIST:
            assert elem.symbol in ELEMENTS
            assert ELEMENTS[elem.symbol] is elem

    def test_symbol_to_id_roundtrip(self):
        for sym, idx in SYMBOL_TO_ID.items():
            assert ELEMENTS_LIST[idx].symbol == sym

    def test_symbol_to_id_covers_all_elements(self):
        assert set(SYMBOL_TO_ID.keys()) == set(ELEMENTS.keys())

    def test_element_fields_are_physical(self):
        for elem in ELEMENTS_LIST:
            assert elem.Z > 0
            assert elem.mass > 0
            assert elem.covalent_radius > 0
            assert 0.0 <= elem.bond_dissociation_self    # kJ/mol, ≥0
            assert 0.0 <= elem.ionization_energy         # eV
            r, g, b = elem.color
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

    def test_noble_gases_have_no_self_bond(self):
        """No covalent self-bond → D(X-X) = 0 by definition for noble gases."""
        for sym in ('He', 'Ne', 'Ar'):
            assert ELEMENTS[sym].bond_dissociation_self == 0.0

    def test_max_bonds_non_negative(self):
        for elem in ELEMENTS_LIST:
            assert elem.max_bonds >= 0

    def test_noble_gases_have_zero_bonds(self):
        for sym in ('He', 'Ne', 'Ar'):
            assert ELEMENTS[sym].max_bonds == 0

    def test_hydrogen_one_bond(self):
        assert ELEMENTS['H'].max_bonds == 1

    def test_carbon_four_bonds(self):
        assert ELEMENTS['C'].max_bonds == 4


class TestGetHelper:
    def test_get_known_element(self):
        elem = get('H')
        assert elem is not None
        assert elem.symbol == 'H'

    def test_get_unknown_returns_none(self):
        assert get('Xx') is None


class TestFusionProduct:
    def test_h_h_gives_deuterium(self):
        assert fusion_product('H', 'H') == 'D'

    def test_h_h_order_independent(self):
        assert fusion_product('H', 'H') == fusion_product('H', 'H')

    def test_d_d_gives_helium(self):
        assert fusion_product('D', 'D') == 'He'

    def test_he_he_gives_beryllium(self):
        assert fusion_product('He', 'He') == 'Be'

    def test_triple_alpha_completes(self):
        # He + Be → C
        assert fusion_product('He', 'Be') == 'C'
        assert fusion_product('Be', 'He') == 'C'

    def test_alpha_capture_chain(self):
        chain = [('C', 'He', 'O'), ('O', 'He', 'Ne'), ('Ne', 'He', 'Mg'),
                 ('Mg', 'He', 'Si'), ('Si', 'He', 'S')]
        for a, b, product in chain:
            assert fusion_product(a, b) == product, f'{a}+{b} should give {product}'

    def test_silicon_burning_endpoint(self):
        assert fusion_product('Si', 'Si') == 'Fe'

    def test_no_reaction_returns_none(self):
        assert fusion_product('Fe', 'Fe') is None
        assert fusion_product('H', 'Fe') is None

    def test_all_products_are_in_element_table(self):
        from sim.elements import FUSION_REACTIONS
        for product in FUSION_REACTIONS.values():
            assert product in ELEMENTS, f'Fusion product {product!r} missing from ELEMENTS'
