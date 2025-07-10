import { useState,useEffect } from 'react'
import './App.css'

function App() {
  const [data, setData] = useState();
  const [feats, setFeats] = useState();
  const [homeworld, setHome] = useState("Tatooine");
  const [unitType, setUnitType] = useState("stormtrooper");
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    console.log(`home:${homeworld}, unit_type:${unitType}`)
    fetchSetData();
    setLoading(false)
  }, [homeworld,unitType,feats])

  // ideally define these in a database and call from an api
  const list_homeworld = ['Tatooine', 'Alderaan', 'Naboo', 'Kashyyyk', 
                          'Stewjon', 'Eriadu', 'Corellia', 'Rodia', 
                          'Bestine IV', 'Dagobah', 'Trandosha', 'Socorro', 
                          'Mon Cala', 'Chandrila', 'Sullust', 'Toydaria', 
                          'Malastare','Dathomir', 'Ryloth', 'Aleen Minor', 
                          'Vulpter', 'Troiken', 'Tund', 'Haruun Kal', 'Cerea', 
                          'Glee Anselm', 'Iridonia', 'Tholoth', 'Iktotch', 
                          'Quermia', 'Dorin', 'Champala', 'Mirial', 'Serenno', 
                          'Concord Dawn', 'Zolan', 'Ojom', 'Skako', 'Muunilinst', 
                          'Shili', 'Kalee', 'Umbara']
  
  const list_units = ["stormtrooper", "tie_fighter", "at-st", "x-wing","resistance_soldier", "at-at", "tie_silencer", "unknown" ]

  async function fetchSetData() {
    const res = await fetch(`http://localhost:5000/predict?homeworld=${homeworld}&unit_type=${unitType}`,{method:"POST"});
    const result = await res.json();
    setData(result)
  }

  async function fetchFeats() {
    const res = await fetch(`http://localhost:5000/features`);
    const result = await res.json();
    setFeats(result)
  }

  if (loading) return <p>Loading data...</p>;
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name === 'homeworld') {
      setHome(value);
    } else if (name === 'unit_type') {
      setUnitType(value);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault(); // Prevents default form submission (page reload)
    // Your custom logic for handling form submission
    // console.log(`city:${city}, units:${units}`);
    // Access form data, perform validation, make API calls, etc.
  };

  return (
    <>
      <h1>star wars</h1>
      <form onSubmit={handleSubmit}>
        <label>Homeworld:</label><br />
        <select name="homeworld" id="homeworld" onChange={handleChange}>
          {list_homeworld.map((x) => (
              <option key={x} value={x}>{x}</option>
          ))}
        </select><br />
        <label>Unit:</label><br />
        <select name="unit_type" id="unit_type" onChange={handleChange}>
          {list_units.map((x) => (
            <option key={x} value={x}>{x}</option>
        ))}
        </select>
        <br /><br />
        {/* <input type="submit" value="Submit" /> */}
      </form>
      <p>{JSON.stringify(data, null, 2)}</p>
      <input type='button' value='get feaatures' onClick={() => (fetchFeats())}/>
      <div>
        <p>{JSON.stringify(feats, null, 2)}</p>
      </div>
    </>
  )
}

export default App
