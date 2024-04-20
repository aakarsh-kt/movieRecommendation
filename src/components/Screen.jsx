// import React from "react";
// import {
//   TextField,
//   Select,
//   FormControl,
//   MenuItem,
//   InputLabel,
// } from "@mui/material";

// import { Button } from "antd";
// export default function () {
//   const choices = [
//     "Content Based Collabrative Filtering",
//     "Collabrative Filtering",
//   ];
//   const [choice, setChoice] = React.useState("");
//   const [subChoice, setSubChoice] = React.useState("");
//   const subChoices1 = ["SVM", "Naive Bayes"];
//   const moreSubChoices1 = ["Linear", "Polynomial", "RBF"];
//   const subChoices2 = ["KNN", "KMeans", "LR"];
//   const [movie, setMovie] = React.useState("");
//   function handleChange(event) {
//     console.log(event.target.value);
//     setChoice(event.target.value);
//   }
//   function handleSubChange(event) {
//     console.log(event.target.value);
//     setSubChoice(event.target.value);
//   }
//   function SubChoice1() {
//     return (
//       <div className="flex--width">
//         <FormControl fullWidth className="form">
//           <InputLabel id="demo-simple-select-label">Choose</InputLabel>
//           <Select
//             labelId="demo-simple-select-label"
//             id="demo-simple-select"
//             value={subChoice}
//             label="Choose"
//             onChange={handleSubChange}
//           >
//             <MenuItem value={choices[0]}>
//               Content Based Collabrative Filtering
//             </MenuItem>
//             <MenuItem value={choices[1]}>Collabrative Filtering</MenuItem>
//           </Select>
//         </FormControl>
//       </div>
//     );
//   }
//   function SubChoice2() {
//     return (
//       <div>
//         <FormControl fullWidth className="form">
//           <InputLabel>Choose</InputLabel>
//           <Select
//             //   labelId="demo-simple-select-label"
//             //   id="demo-simple-select"
//             value={subChoice}
//             label="Choose"
//             onChange={handleSubChange}
//           >
//             <MenuItem value={choices[0]}>
//               Content Based Collabrative Filtering
//             </MenuItem>
//             <MenuItem value={choices[1]}>Collabrative Filtering</MenuItem>
//           </Select>
//         </FormControl>
//       </div>
//     );
//   }
//   return (
//     <div className="flex--column flex--gap">
//       <TextField
//         label="Enter the Movie!!!"
//         onChange={(event) => {
//           setMovie(event.target.value);
//         }}
//         className="text--field"
//       />
//       <Button
//         label="Submit"
//         type="primary"
//         onClick={() => {
//           console.log(movie);
//         }}
//       >
//         Submit
//       </Button>

//       <h1>Screen</h1>
//       <FormControl fullWidth className="form">
//         <InputLabel>Choose</InputLabel>
//         <Select
//           //   labelId="demo-simple-select-label"
//           //   id="demo-simple-select"
//           value={choice}
//           label="Choose"
//           onChange={handleChange}
//         >
//           <MenuItem value={choices[0]}>
//             Content Based Collabrative Filtering
//           </MenuItem>
//           <MenuItem value={choices[1]}>Collabrative Filtering</MenuItem>
//         </Select>
//       </FormControl>
//       {choice === "Content Based Collabrative Filtering" && <SubChoice1 />}
//       {choice === "Collabrative Filtering" && <SubChoice2 />}
//     </div>
//   );
// }
import React, { useState } from 'react';

export default function() {
  const [movieName, setMovieName] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');

  const fetchRecommendations = () => {
    fetch(`http://localhost:5000/recommend?movie=${encodeURIComponent(movieName)}`)
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          setError(data.error);
          setRecommendations([]);
        } else {
          setRecommendations(data);
          setError('');
        }
      })
      .catch(error => {
        console.error('Error:', error);
        setError('Failed to fetch data. Please try again later.');
        setRecommendations([]);
      });
  };

  const handleInputChange = (event) => {
    setMovieName(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();  // Prevent the form from reloading the page
    fetchRecommendations();
  };

  return (
    <div>
      <h1>Movie Recommendation</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={movieName}
          onChange={handleInputChange}
          placeholder="Enter a movie name"
        />
        <button type="submit">Get Recommendations</button>
      </form>
      {error && <p>Error: {error}</p>}
      <ul>
        {recommendations.map((movie, index) => (
          <li key={index}>
            {movie.title} - {movie.genres}
          </li>
        ))}
      </ul>
    </div>
  );
}


