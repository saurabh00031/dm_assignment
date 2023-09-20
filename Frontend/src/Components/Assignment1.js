import axios from "axios";
import React, { useEffect, useState } from 'react';

// import './Css/Assignment1.css';
const Assignment1 = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('http://localhost:8000/assignment1')
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
    <div>
      {loading ? (
        <p>Loading...</p>
      ) : data ? (
        <div className="container mt-5">
          <h4><u>NAME : {data.name}</u></h4>
          <div className="table-responsive">
            <table className="table table-striped mt-3">
              <thead>
                <tr>
                  <th>Attribute</th>
                  <th>Mean</th>
                  <th>Median</th>
                  <th>Mode</th>
                  <th>Variance</th>
                  <th>Standard Deviation</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(data.mode).map((attribute) => (
                  <tr key={attribute}>
                    <td>{attribute}</td>
                    <td>{data.mean[attribute]}</td>
                    <td>{data.median[attribute]}</td>
                    <td>{data.mode[attribute]}</td>
                    <td>{data.variance[attribute]}</td>
                    <td>{data.std[attribute]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <p>No data available.</p>
      )}
    </div>
  );
}

export default Assignment1;
