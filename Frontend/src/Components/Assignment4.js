import React, { useEffect, useState } from 'react';
import axios from 'axios';
// import './Css/Assignment4.css'; // Import the CSS file

const Assignment4 = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get('http://localhost:8000/assignment4')
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
    <div className="assignment4-container">
      {loading ? (
        <p className="loading-text">Loading...</p>
      ) : data ? (
        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th colSpan="2">Name Of File:</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan="2">{data.name}</td>
              </tr>
              <tr>
                <th>Rules:</th>
                <td>{data.rules}</td>
              </tr>
              <tr>
                <th>Accuracy:</th>
                <td>{data.accuracy}</td>
              </tr>
              <tr>
                <th>Coverage:</th>
                <td>{data.coverage}</td>
              </tr>
              <tr>
                <th>Toughness:</th>
                <td>{data.toughness}</td>
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

export default Assignment4;
