import Header from "../components/Header";
import "../styles/ScheduleDetail.css";

import { IoBookmarkSharp } from "react-icons/io5";

import {useState, useEffect, useMemo} from "react";
import kingoMIcon from "../assets/images/kingo-M.png";
import sort from "../assets/images/sort.png";


import React, { Component } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import axios from "axios";

function ScheduleDetail({}){
    
    const location = useLocation();

    const navigate = useNavigate();

    console.log('detail', location.state)
    const [notices, setNotices] = useState(location.state?.notice ? location.state.notice : []);
    const [sortedNotices, setSortedNotices] = useState(notices.sort((a, b) => a.read - b.read));
    const title = location.state?.title ? location.state.title : "";
    const selectedFavorites = location.state?.selectedFavorites ? location.state.selectedFavorites : {};
    const preMark = location.state?.selectedMark ? location.state.selectedMark : {};


    const access_token = localStorage.getItem('access_token');
    console.log(access_token); 
    const access_token_with_header = "Bearer " + access_token;
    const [message, setMessage] = useState("");
    // const title = '학사과정 조기졸업/석사과정 수업연한 단축/석박사통합과정 조기수료·이수포기 신청';
    // const notice = ['2024년 여름 전체 학위수여식 참석(신청) 안내(졸업생/축하객, 신청일: 학사 8.13./석사 8.14.)',
    //     '2024학년도 2학기 학사과정 조기졸업 신청 안내',
    //     '2024년 금신사랑장학생 선발 안내',
    //   ]

    const [reads, setReads] = useState(notices.reduce((acc, item) => {
        if (item.read) {
          acc[item.id] = true; // d.noti_id를 키로, true를 값으로 설정
        } else{
          acc[item.id] = false;
        }
        return acc; // 누적된 딕셔너리 반환
      }, {}));


    useEffect(()=>{
        const fetchData = async()=>{
            try {
                console.log("Fetching schedule-related notices...");
                const response = await axios.post(`http://127.0.0.1:5000/user/schedule`, 
                  { title: title },
                  { headers: { Authorization: access_token_with_header } }
                  );
        
                if (response.data.msg === "get schedule-related notices success") {
                  const relatedNotice = response.data.data.map((item) => ({
                    id: item.noti_id,
                    title: item.title,
                    url: item.url,
                    read: item.read,
                    isBookMarked: item.isBookMarked,
                  }));
        
                  setNotices(relatedNotice);
                  setSortedNotices(relatedNotice.sort((a, b) => a.read - b.read));

                  const updatedRead = relatedNotice.reduce((acc, item) => {
                    if (item.read) {
                      acc[item.id] = true; // d.noti_id를 키로, true를 값으로 설정
                    } else{
                      acc[item.id] = false;
                    }
                    return acc; // 누적된 딕셔너리 반환
                  }, {});
                  setReads(updatedRead);

                  setMessage(response.data.msg);
                } else {
                  setMessage(response.data.msg);
                }
              } catch (error) {
                if (error.response) {
                  setMessage(error.response.data.msg);
                } else {
                  setMessage("An error occurred while connecting to the server.");
                }
              }
        }

        fetchData();
    }, []);



    console.log("notices", notices);
    const [content, setContent] = useState("");

    const [selectedMark, setSelectedMark] = useState(preMark ? preMark : {});
    const toggleMark = async (noticeIdx, e, noticeID) => {
        setSelectedMark((prevState) => ({
          ...prevState,
          [noticeIdx]: !prevState[noticeIdx], // 해당 이벤트의 별표 상태를 반전시킴
        }));

        e.preventDefault();

        if(selectedMark[noticeIdx]){
            try {
                const response = await axios.post(`http://127.0.0.1:5000/user/noti/${noticeID}`,
                    { scrap: 0 },
                    { headers: { Authorization: access_token_with_header } }
                );
            console.log("response.data", response.data);
                // 서버로부터 받은 응답 처리
            if (response.data.msg === "delete success") {
                console.log("response data", response.data);
                setMessage(response.data.msg); // "register keyword successful"
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
        }
        else{
            try {
                const response = await axios.post(`http://127.0.0.1:5000/user/noti/${noticeID}`,
                    { scrap: 1 },
                    { headers: { Authorization: access_token_with_header } }
                );
                console.log("response.data", response.data);
                    // 서버로부터 받은 응답 처리
                if (response.data.msg === "scrap success") {
                    console.log("response data", response.data);
                    setMessage(response.data.msg); // "register keyword successful"
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
        }
        

      };

    const eventClickHandler = async (notice, idx, e) => {
        console.log("click", notice, idx);
        var propsNotice = sortedNotices;
        const updatedNotices = sortedNotices.map((notice, i) =>
            i === idx ? {title: notice.title, read: 1, url: notice.url} : notice // 클릭한 항목의 두 번째 요소를 1로 변경
        );
        setSortedNotices(updatedNotices.sort((a, b) => a.read - b.read));
        console.log("propsNotice",propsNotice, sortedNotices)

        e.preventDefault();
        
        try {
          console.log("Edit schedule-related notices...");
          const response = await axios.patch(`http://127.0.0.1:5000/user/noti/${notice.id}`, 
            {  update: "read"},
            { headers: { Authorization: access_token_with_header } }
        );
  
          if (response.data.msg === "edit related notice read success") {
            
            setMessage(response.data.msg);
          } else {
            setMessage(response.data.msg);
          }
        } catch (error) {
          if (error.response) {
            setMessage(error.response.data.msg);
          } else {
            setMessage("An error occurred while connecting to the server.");
          }
        }

        // return <Navigate to="/ScheduleDetail" title={arg.event.title} notice={arg.event.notice} />;
        navigate("/schedule/detail/notice", {state:{scheduleTitle: title, title: notice.title, noticeURL: notice.url, notices: updatedNotices, selectedMark: selectedMark, page:"scheduleDetail", access_token:access_token }});
    };

    const [isAscending, setIsAscending] = useState(true); // 오름차순/내림차순 상태

    // 정렬 핸들러
    const handleSort = () => {
        const sorted = [...sortedNotices].sort((a, b) => {
        if (isAscending) return b.title.localeCompare(a.title); // 내림차순
        return a.title.localeCompare(b.title); // 오름차순
        });
        setSortedNotices(sorted);
        setIsAscending(!isAscending); // 정렬 상태 토글
    };

    const markAllRead = () => {
        console.log("markAllRead")
        setSortedNotices((prevNotices) => {
            const updatedNotices = prevNotices.map((notice, i) => ({title: notice.title, read: 1, url: notice.url}));            // 변경된 배열 다시 정렬            
            console.log('updatedNotices',updatedNotices);
            return updatedNotices;
        });
    }


    // 공지 검색 처리
    const filteredNotices = sortedNotices.filter((notice) =>
        notice.title.includes(content) // 제목에 검색어 포함 여부 확인
    );

    return(
       <div className="relative w-full h-full bg-white">
            <Header page="ScheduleDetail" selectedFavorites={selectedFavorites} />
            <div className="line"></div>

            {/* <div className="flex justify-between space-x-1 text-center home_tabs bg-white">
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-blue-900 id_tab cursor-pointer">신분증</button>
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-zinc-300 kingo_tab cursor-pointer">KINGO ⓘ</button>
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-zinc-300 favor_tab cursor-pointer">즐겨찾기</button>
            </div> */}

            <div id="detail-container" className="h-fit m-3 mb-5 rounded-lg p-5 pt-3">
                <div className="flex flex-row h-full">
                    {/* <div className="flex-none pt-2 pr-2 table_of_contents">
                        <img src={kingoMIcon} width={"50px"} height={"75px"}/>
                        <div className="iconTitle font-bold text-l relative bottom-3">KINGO-M</div>

                        <div className="flex flex-col text-md">
                            <button className="idx_btn selected text-md">알림 공지</button>
                            <button className="idx_btn text-md">스크랩 공지</button>
                        </div>
                    </div> */}
                    <div className="flex-auto w-full">
                        <div className="pt-2 flex w-full overflow-hidden">
                            <div className="flex-auto related_notice_title font-bold text-s ellipsText text-left 
                                overflow-hidden whitespace-nowrap text-ellipsis">{title}</div>
                            {/* <div className="flex-none text-left font-bold">...</div> */}
                        </div>
                        <div className="pb-2 related_notice_title font-bold text-s w-full block text-left"> 관련 공지 </div>

                        <div className='sb'>
                            <input className="searchbar" placeholder="검색어를 입력하세요." type="text" name="content" value={content} onChange={(e)=>setContent(e.target.value)} />
                            <button className="searchbar_btn" type="submit">검색</button>
                        </div>
                        
                        <div className="mt-9 line"></div>

                        <div className="flex flex-row justify-between mt-2 sub_func w-full">
                            <div className="flex-none text-xs mt-1 pt-1 cursor-pointer" onClick={handleSort}>
                                <span className="inline-block float-left" >정렬</span>
                                <img className="inline-block float-left" src={sort} width="17px" height="17px" />
                            </div>
                            <div className="flex-auto"></div>
                            <div className="flex-none">
                                {/* <button className="alarm_del">공지 삭제</button> */}
                                <button className="mark_read justify-items-end" onClick={markAllRead}>Mark All Read</button>
                            </div>
                        </div>

                        {(content ? filteredNotices.length > 0 : sortedNotices.length > 0) && (<div className="flex flex-col mt-3 notices">
                            {(content ? filteredNotices:sortedNotices).map((notice, idx)=>{
                                const isMarked = selectedMark[idx];
                                return(
                                    <div className="w-full">
                                        <div className="flex flex-row gap-1 justify-between notice w-full cursor-pointer" style={{backgroundColor: reads[notice.id] ? "lightgray" : "darkgray"}} key={idx}
                                        onClick={(e)=>{eventClickHandler(notice, idx, e)}}>
                                            <div className="flex-auto bg-inherit m-auto text-left align-middle h-fit">{notice.title}</div>
                                            <div className="flex-none event-favorite-star align-middle bg-inherit m-0" 
                                            style={{boxSizing:"border-box", backgroundColor:"inherit", color: isMarked ? 'red' : 'gray'}}
                                            onClick={(event)=>{event.stopPropagation();toggleMark(idx, event, notice.id)}}>
                                            <IoBookmarkSharp className="bg-inherit align-middle h-full m-auto pt-1" />
                                            </div> 
                                        </div>
                                        
                                    </div>
                                )
                            }) }
                        </div> )}
                        
                    </div>
                </div>
            </div>
              

        </div>

    )

}

export default ScheduleDetail;