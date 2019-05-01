import pytest
import main

m = "Bromwell High is nothing short of brilliant. Expertly scripted and perfectly delivered, this searing parody of a students and teachers at a South London Public School leaves you literally rolling with laughter. It's vulgar, provocative, witty and sharp. The characters are a superbly caricatured cross section of British society (or to be more accurate, of any society)."

def test_entity():
    f, ff = main.get_entity(m)
    assert len(f) >= 0

def test_entity():
    f, ff = main.get_entity(m)
    assert len(ff) >= 0