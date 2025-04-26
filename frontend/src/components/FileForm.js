import React, { useState, useEffect, useRef } from "react";

function FileForm () {
  const [file, setFile] = useState(null);
  const [uploadState, setUploadState] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (uploadState) {
      const timeout = setTimeout(() => setUploadState(null), 3000);
      return () => clearTimeout(timeout);
    }
  }, [uploadState]);

  const handleFileInputChange = (event) => {
    setFile(event.target.files[0])
  }

  const handleSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append('file_upload', file);

    try{
      const endpoint = "http://localhost:8000/uploadflie/";
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData
      })

      if (response.ok){
        console.log("File uploaded successfully");
        setUploadState("File uploaded successfully!");
      } else {
        console.error("File upload failed");
        setUploadState("File upload failed");
      }
    } catch (error) {
      console.log("Error uploading file:", error)
      setUploadState("Error uploading file");
    }

    setFile(null);

    if(fileInputRef.current){
      fileInputRef.current.value = "";
    }
  }

  return (
  <div className="flex flex-col items-start mb-2">
    <div className="max-w-md bg-white p-2 rounded-lg w-full">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input 
          type="file"
          onChange={handleFileInputChange}
          ref={fileInputRef}
          accept="application/msword, application/vnd.openxmlformats-officedocument.wordprocessingml.document, text/html, application/json, text/markdown, application/pdf, text/plain" 
          className="relative m-0 block w-full min-w-0 flex-auto cursor-pointer rounded border border-solid border-neutral-300 bg-clip-padding px-3 py-[0.32rem] text-xs font-normal text-neutral-700 transition duration-300 ease-in-out file:-mx-3 file:-my-[0.32rem] file:cursor-pointer file:overflow-hidden file:rounded-none file:border-0 file:border-solid file:border-inherit file:bg-neutral-100 file:px-3 file:py-[0.32rem] file:text-neutral-700 file:transition file:duration-150 file:ease-in-out file:[border-inline-end-width:1px] file:[margin-inline-end:0.75rem] hover:file:bg-blue-600 focus:border-primary focus:text-neutral-700 focus:shadow-te-primary focus:outline-none dark:border-neutral-600 dark:text-neutral-500 dark:file:bg-neutral-700 dark:file:text-neutral-100 dark:focus:border-primary"
        />
        <button 
          type="submit" 
          className="dark:bg-neutral-700 text-white text-xs font-normal px-4 py-1.5 rounded hover:bg-blue-600 transition"
        >
          Upload
        </button>

        <div className="flex-shrink-0 w-1 ml-2">
          <p className={`text-xs whitespace-nowrap ${uploadState ? 'text-green-500' : 'text-red-500'}`}>
            {uploadState}
          </p>
        </div>
      </form>
    </div>
  </div>
  );
}

export default FileForm