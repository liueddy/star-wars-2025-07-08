from pydantic import BaseModel, PositiveInt, NonNegativeInt, NonNegativeFloat
from typing import Literal


class SWDantic(BaseModel):
    """Validates a user input for the Star Wars API."""

    # valid unit types based on generate_data.py
    unit_type: Literal["stormtrooper", "tie_fighter", "at-st", "x-wing",
            "resistance_soldier", "at-at", "tie_silencer", "unknown"]
    
    # valid homeworlds based on home_worlds.json
    homeworld: Literal['Tatooine', 'Alderaan', 'Naboo', 'Kashyyyk', 
                       'Stewjon', 'Eriadu', 'Corellia', 'Rodia', 
                       'Bestine IV', 'Dagobah', 'Trandosha', 'Socorro', 
                       'Mon Cala', 'Chandrila', 'Sullust', 'Toydaria', 
                       'Malastare','Dathomir', 'Ryloth', 'Aleen Minor', 
                       'Vulpter', 'Troiken', 'Tund', 'Haruun Kal', 'Cerea', 
                       'Glee Anselm', 'Iridonia', 'Tholoth', 'Iktotch', 
                       'Quermia', 'Dorin', 'Champala', 'Mirial', 'Serenno', 
                       'Concord Dawn', 'Zolan', 'Ojom', 'Skako', 'Muunilinst', 
                       'Shili', 'Kalee', 'Umbara']
    
    # this kinda does not matter, but if you want to specify, 
    # this class will not stop you from inputing.
    location_x: float = 0
    location_y: float = 0
    destination_x: float = 0
    destination_y: float = 0