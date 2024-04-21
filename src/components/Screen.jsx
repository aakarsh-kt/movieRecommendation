import React from "react";
import {
  TextField,
  Select,
  FormControl,
  MenuItem,
  InputLabel,
} from "@mui/material";

import { Button } from "antd";
export default function () {
  const choices = [
    "Content Based Filtering",
    "Collabrative Filtering",
  ];
  const [currModel, setCurrModel] = React.useState("");
  const [movieName, setMovieName] = React.useState("");
  const [recommendations, setRecommendations] = React.useState([]);
  const [error, setError] = React.useState("");
  const [choice, setChoice] = React.useState("");
  const [subChoice, setSubChoice] = React.useState("");
  const subChoices1 = ["svm", "Naive Bayes"];
  const [subsubchoice, setSubsubChoice] = React.useState("");
  const moreSubChoices1 = ["Linear", "Polynomial", "RBF"];
  const subChoices2 = ["knn", "kmeans", "LR"];
  const fetchRecommendations = (model) => {
    fetch(
      `http://localhost:5000/recommend?movie=${encodeURIComponent(
        movieName
      )}&model=${model}`
    )
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          setError(data.error);
          setRecommendations([]);
        } else {
          setRecommendations(data);
          setError("");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        setError("Failed to fetch data. Please try again later.");
        setRecommendations([]);
      });
  };

  //   const fetchRecommendations = () => {
  //     fetch(
  //       `http://localhost:5000/recommend?movie=${encodeURIComponent(movieName)}`
  //     )
  //       .then((response) => response.json())
  //       .then((data) => {
  //         if (data.error) {
  //           setError(data.error);
  //           setRecommendations([]);
  //         } else {
  //           setRecommendations(data);
  //           setError("");
  //         }
  //       })
  //       .catch((error) => {
  //         console.error("Error:", error);
  //         setError("Failed to fetch data. Please try again later.");
  //         setRecommendations([]);
  //       });
  //   };

  const handleInputChange = (event) => {
    setMovieName(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault(); // Prevent the form from reloading the page
    fetchRecommendations(currModel);
  };
  function handleChange(event) {
    console.log(event.target.value);
    setChoice(event.target.value);
  }
  function handleSubChange(event) {
    console.log(event.target.value);
    setSubChoice(event.target.value);
  }
  function handleSubSubChange(event) {
    console.log(event.target.value);
    setSubsubChoice(event.target.value);
  }
  function handleApply() {
    setCurrModel(subChoice);
    if (subChoice === "svm" && subsubchoice === "Linear") {
      setCurrModel("svm_linear");
    } else if (subChoice === "SVM" && subsubchoice === "Polynomial") {
      setCurrModel("SVM_Polynomial");
    } else if (subChoice === "SVM" && subsubchoice === "RBF") {
      setCurrModel("SVM_RBF");
    }
    console.log(currModel);
  }
  function SubChoice1() {
    return (
      <div className="flex--width">
        <FormControl className="form select">
          <InputLabel id="demo-simple-select-label">Choose</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={subChoice}
            label="Choose"
            onChange={handleSubChange}
          >
            <MenuItem value={subChoices1[0]}>SVM</MenuItem>
            <MenuItem value={subChoices1[1]}>Naive Bayes</MenuItem>
          </Select>
        </FormControl>
      </div>
    );
  }
  function SubChoice2() {
    // const [moreSubChoice, setMoreSubChoice] = React.useState(["KNN"]);
    return (
      <div>
        <FormControl fullWidth className="form select">
          <InputLabel>Choose</InputLabel>
          <Select
            //   labelId="demo-simple-select-label"
            //   id="demo-simple-select"
            value={subChoice}
            label="Choose"
            onChange={handleSubChange}
          >
            <MenuItem value={subChoices2[0]}>KNN</MenuItem>
            <MenuItem value={subChoices2[1]}>KMeans</MenuItem>
            <MenuItem value={subChoices2[2]}>LR</MenuItem>
          </Select>
        </FormControl>
      </div>
    );
  }
  function SubChoice3() {
    // const [moreSubChoice, setMoreSubChoice] = React.useState(["KNN"]);
    return (
      <div>
        <FormControl fullWidth className="form select">
          <InputLabel>Choose</InputLabel>
          <Select
            //   labelId="demo-simple-select-label"
            //   id="demo-simple-select"
            value={subsubchoice}
            label="Choose"
            onChange={handleSubSubChange}
          >
            <MenuItem value={moreSubChoices1[0]}>Linear</MenuItem>
            <MenuItem value={moreSubChoices1[1]}>Polynomial</MenuItem>
            <MenuItem value={moreSubChoices1[2]}>RBF</MenuItem>
          </Select>
        </FormControl>
      </div>
    );
  }
  return (
    <div className="flex--column flex--gap">
      <TextField
        label="Enter the Movie!!!"
        onChange={(event) => {
          setMovieName(event.target.value);
        }}
        className="text--field"
      />
      <Button label="Submit" type="primary" onClick={handleSubmit}>
        Submit
      </Button>

      <h1>Rakshit</h1>
      <FormControl className="form select">
        <InputLabel>Choose</InputLabel>
        <Select
          //   id="demo-simple-select"
          value={choice}
          label="Choose"
          onChange={handleChange}
        >
          <MenuItem value={choices[0]}>
            Content Based Filtering
          </MenuItem>
          <MenuItem value={choices[1]}>Collabrative Filtering</MenuItem>
        </Select>
      </FormControl>
      <div className="flex--width ">
        {choice === "Content Based Filtering" && <SubChoice1 />}
        {choice === "Collabrative Filtering" && <SubChoice2 />}

        {subChoice === "svm" && <SubChoice3 />}
      </div>
      <Button type="primary" onClick={handleApply}>
        Apply
      </Button>
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
// import React, { useState } from 'react';

// export default function() {
//   const [movieName, setMovieName] = useState('');
//   const [recommendations, setRecommendations] = useState([]);
//   const [error, setError] = useState('');

//   const fetchRecommendations = () => {
//     fetch(`http://localhost:5000/recommend?movie=${encodeURIComponent(movieName)}`)
//       .then(response => response.json())
//       .then(data => {
//         if (data.error) {
//           setError(data.error);
//           setRecommendations([]);
//         } else {
//           setRecommendations(data);
//           setError('');
//         }
//       })
//       .catch(error => {
//         console.error('Error:', error);
//         setError('Failed to fetch data. Please try again later.');
//         setRecommendations([]);
//       });
//   };

//   const handleInputChange = (event) => {
//     setMovieName(event.target.value);
//   };

//   const handleSubmit = (event) => {
//     event.preventDefault();  // Prevent the form from reloading the page
//     fetchRecommendations();
//   };

//   return (
//     <div>
//       <h1>Movie Recommendation</h1>
//       <form onSubmit={handleSubmit}>
//         <input
//           type="text"
//           value={movieName}
//           onChange={handleInputChange}
//           placeholder="Enter a movie name"
//         />
//         <button type="submit">Get Recommendations</button>
//       </form>
//       {error && <p>Error: {error}</p>}
//       <ul>
//         {recommendations.map((movie, index) => (
//           <li key={index}>
//             {movie.title} - {movie.genres}
//           </li>
//         ))}
//       </ul>
//     </div>
//   );
// }
