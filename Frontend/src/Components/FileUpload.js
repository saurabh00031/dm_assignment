import React, { useState } from 'react';
import axios from 'axios';
import './Css/FileUpload.css'; // Import the CSS file

function FileUpload() {
  const [file, setFile] = useState(null);
  const [name, setName] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleNameChange = (e) => {
    setName(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file || !name) {
      alert('Please select a file and enter a name.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);

    try {
      const response = await axios.post(
        'http://localhost:8000/api/upload-csv/',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      // Handle the response here
      console.log('File uploaded successfully:', response.data);
    } catch (error) {
      // Handle errors
      console.error('File upload error:', error);
    }
  };

  return (
    <div className="container mt-5">
      <form onSubmit={handleSubmit} className="file-upload-form">
      <h3 className="text-center">Data Mining Tool</h3>
        <div className="mb-3">
          <label htmlFor="formFile" className="form-label">
            Upload CSV File
          </label>
          <input
            className="form-control"
            type="file"
            onChange={handleFileChange}
            accept=".csv"
            id="formFile"
          />
        </div>

        <div className="mb-3">
          <label htmlFor="exampleFormControlInput1" className="form-label">
            File Name
          </label>
          <input
            type="text"
            value={name}
            placeholder="Name"
            className="form-control"
            onChange={handleNameChange}
            id="exampleFormControlInput1"
          />
        </div>

        <div className="text-center">
          <button className="btn btn-primary" type="submit">
            Upload
          </button>
        </div>
      </form>
    </div>
  );
}

export default FileUpload;
