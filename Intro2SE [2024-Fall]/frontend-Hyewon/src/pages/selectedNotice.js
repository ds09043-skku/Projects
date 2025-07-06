import Header from "../components/Header";
import "../styles/ScheduleDetail.css";

import ExternalPage from "../components/ExternalPage";

import React, { Component } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function SelectedNotice(){
    const location = useLocation();   
    const access_token = localStorage.getItem('access_token');
    console.log(access_token); 
    console.log(location.state)
    const url = location.state.noticeURL ? location.state.noticeURL : [];
    const title = location.state.title ? location.state.title : "";
    const scheduleTitle = location.state.scheduleTitle ? location.state.scheduleTitle : "";
    const notice = location.state.notices ? location.state.notices : "";
    const selectedMark = location.state.selectedMark ? location.state.selectedMark : {};
    const prevPage = location.state.page ? location.state.page : "";

    return(
       <div className="flex flex-col relative w-full h-full bg-white">
            <div className="flex-none">
              <Header className="" page="selectedNotice" title={title} notice={notice} scheduleTitle={scheduleTitle} selectedMark={selectedMark} prevPage={prevPage}/>
              <div className="line"></div>
            </div>
            
            <div className="flex-auto" style={{ width: '100vw', height: '100%', maxWidth:"400px" }}>
              <ExternalPage url={url} />
            </div>

        </div>

    )

}

export default SelectedNotice;