import "../styles/Login.css";
import kingoMIcon from "../assets/images/kingoMIcon.png";

import React, { Component, useEffect, useState } from 'react';
import axios from "axios";
import { useLocation, useNavigate } from 'react-router-dom';

function Login(){
  const access_token = localStorage.getItem('access_token');
console.log(access_token); 
    const navigate = useNavigate();

    const [id, setId] = useState("");
    const [pw, setPw] = useState("");
    const [logKeep, setLogKeep] = useState(false);

    const [message, setMessage] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault(); // 폼 제출 시 페이지 새로고침 방지
        console.log("id", id);
        console.log("pw", pw);
        try {
          const response = await axios.post("http://127.0.0.1:5000/user/login", {
            id: id,
            pw: pw,
          });
          console.log("response", response);
            // 서버로부터 받은 응답 처리
            if (response.data.msg === "login success") {
            console.log("response data", response.data);
            setMessage(response.data.msg); // "Login successful"
            localStorage.setItem('access_token', response.data.access_token);
            navigate('/home', {state:{access_token:response.data.access_token}})
          } else {
            setMessage(response.data.msg); // "Invalid credentials"
          }
        } catch (error) {
          // 에러 처리
          if (error.response) {
            setMessage(error.response.data.msg); // 서버에서 보낸 에러 메시지
          } else {
            setMessage("An error occurred while connecting to the server.");
          }
        }
      };

    return(
       <div className="flex flex-col relative w-full h-full " style={{backgroundColor:"#f9f9f9"}}>
            <div className="flex-none bg-inherit mt-10 mb-3">
              <img className="bg-inherit" src={kingoMIcon}/>
            </div>
            
            <div className="flex flex-col flex-auto gap-1 bg-inherit pt-5 p-10" style={{ width: '100vw', height: '100%', maxWidth:"400px" }}>
                <input className="flex login-input-bar id" content={id} onChange={(e)=>setId(e.target.value)} placeholder="아이디를 입력하세요"/>
                <input className="flex login-input-bar pw" content={pw} type="password" onChange={(e)=>setPw(e.target.value)} placeholder="비밀번호를 입력하세요"/>
                <div className="flex flex-row gap-0.5 text-xs bg-inherit m-0">
                    <input className="flex-none ml-0.5" type="checkbox" checked={logKeep} onChange={()=>setLogKeep((prev)=>!prev)}/>
                    <div className="flex-auto login-text bg-inherit m-0 text-left">로그인 상태 유지 (자동로그인)</div>
                </div>

                <div className="flex gap-1.5 bg-inherit w-full mt-3">
                    <button className="flex-1 bg-inherit log-btn button-text p-1" onClick={(e)=>handleSubmit(e)}>로그인</button>
                    <button className="flex-1 bg-inherit sim-log-btn button-text p-1" onClick={(e)=>handleSubmit(e)}>간편(생체인증)로그인</button>
                </div>
                
            </div>

        </div>

    )

}

export default Login;