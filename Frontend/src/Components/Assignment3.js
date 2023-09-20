import React, { useEffect, useState } from 'react';
import axios from 'axios';
import image from 'file:///C:/Users/Saurabh/Desktop/DM%20assignments/dm_assignments/backend/static/plot/image.png';
// import './Css/Assignment3.css'; // Import the CSS file

const Assignment3 = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get('http://localhost:8000/assignment3')
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
    <div className="assignment3-container">
      {loading ? (
        <p className="loading-text">Loading...</p>
      ) : data ? (
        <div className="content-container">
          <h1><u>Name Of File: {data.name}</u></h1>
          <table className="data-table">
            <tbody>
              <tr>
                <td>Data:</td>
                <td>{data.text}</td>
              </tr>
              <tr>
                <td>Image:</td>
                <td>
                  <img src={image} alt="no_image" />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      ) : (
        <p className="no-data">No data available.</p>
      )}
    </div>
  );
};

export default Assignment3;
