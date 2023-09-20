import React, { useEffect, useState } from 'react';
import axios from 'axios';
// import './Css/Assignment2.css'; // Import the CSS file

const Assignment2 = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get('http://localhost:8000/assignment2')
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
    <div className="assignment2-container">
      {loading ? (
        <p>Loading...</p>
      ) : data ? (
        <table className="assignment2-table">
          <tbody>
            <tr>
              <td className="assignment2-title" colSpan="2">
                <u>Name Of File: {data.name}</u>
              </td>
            </tr>
            <tr>
              <td>Result:</td>
              <td>{data.result}</td>
            </tr>
            <tr>
              <td>p:</td>
              <td>{data.p}</td>
            </tr>
            <tr>
              {/* <td>chi2:</td> */}
              <td>Chi-Square Test:</td>
              <td>{data.chi2}</td>
            </tr>
            <tr>
              <td>dof:</td>
              <td>{data.dof}</td>
            </tr>
          </tbody>
        </table>
      ) : (
        <p>No data available.</p>
      )}
    </div>
  );
};

export default Assignment2;
