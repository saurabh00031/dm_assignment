import React, { useEffect, useState } from 'react';
import axios from 'axios';
import image1 from 'file:///C:/Users/Saurabh/Desktop/DM assignments/dm_assignments/dm_assignments/dm_assignments/static/ANNplot.png';
import image2 from 'file:///C:/Users/Saurabh/Desktop/DM assignments/dm_assignments/dm_assignments/dm_assignments/static/ANNplot.png';
// import './Css/Assignment5.css'; // Import the CSS file

const Assignment5 = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get('http://localhost:8000/assignment5')
      .then((response) => {
        setData(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Axios error:', error);
        setLoading(false);
      });
  }, []);

  return (
    <div className="assignment5-container">
      {loading ? (
        <p className="loading-text">Loading...</p>
      ) : data ? (
        <div className="content-container">
          <div className="slide-down">
            <h1><u>Name Of File: {data.name}</u></h1>

            <h3>ANN</h3>
            <img src={image1} alt="ANN" />

            <h3>KNN</h3>
            <img src={image2} alt="KNN" />
          </div>
        </div>
      ) : (
        <p className="no-data">No data available.</p>
      )}
    </div>
  );
};

export default Assignment5;
