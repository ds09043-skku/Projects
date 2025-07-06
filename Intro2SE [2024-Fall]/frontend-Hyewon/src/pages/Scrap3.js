import React, { useState, useEffect } from "react";
import "../styles/Scrap.css";
import { useLocation, useNavigate } from "react-router-dom";
import Header from "../components/Header";
import kingoMIcon from "../assets/images/kingo-M.png";
import { IoBookmarkSharp } from "react-icons/io5";
import sort from "../assets/images/sort.png";
import axios from "axios";

function NotificationRelated() {
  
  const navigate = useNavigate();
  const location = useLocation();
  const [currentView, setCurrentView] = useState("main");
  const [message, setMessage] = useState("");

  // 전달받은 공지사항 데이터
  const selectedNotification = location.state?.notification || {};
  const keywordid = location.state?.keywordid ? location.state.keywordid : "";
  const access_token = localStorage.getItem('access_token');
  console.log(access_token); 
  const access_token_with_header = "Bearer " + access_token;

  // 더미 관련 데이터
  const [relatedNotifications, setRelatedNotifications] = useState([]);
  //   { id: 1, title: "[긴급] 2024학년도 2학기 수강신청 관련 안내", isScrapped: false, isRead:1, url:""},
  //   { id: 2, title: "2024학년도 2학기 수강신청 안내", isScrapped: true, isRead:1, url:"https://www.skku.edu/skku/campus/skk_comm/notice01.do?mode=view&articleNo=122612&article.offset=0&articleLimit=10&srSearchVal=2024%ED%95%99%EB%85%84%EB%8F%84+%ED%95%99%EC%82%AC%EA%B3%BC%EC%A0%95+%EA%B2%A8%EC%9A%B8+%EA%B3%84%EC%A0%88%EC%88%98%EC%97%85+%EC%9A%B4%EC%98%81+%EC%95%88%EB%82%B4"},
  //   { id: 3, title: "2024학년도 2학기 수강철회 안내", isScrapped: false, isRead:0, url:""},
  //   { id: 4, title: "석박사통합과정 조기수료·이수포기 신청 안내", isScrapped: false, isRead:0, url:""},
  // ]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log("Fetching related notices...");
        const response = await axios.get(`http://127.0.0.1:5000/user/${keywordid}`, {
            access_token: access_token_with_header
        });

        if (response.data.msg === "get regist keyword notice success") {
          const mappedData = response.data.data.map((item) => ({
            id: item.noti_id,
            title: item.title,
            url: item.url,
            isRead: item.read,
            isScrapped: item.scrap,
          }));
          setRelatedNotifications(mappedData);
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
    };

    fetchData(); // 비동기 함수 호출
  }, []); // 빈 배열: 컴포넌트 마운트 시 한 번만 실행



  const handleMarkAllRead = async (e) => {
    setRelatedNotifications((prev) =>
      prev.map((notif) => ({ ...notif, isRead: false }))
    );

    e.preventDefault(); // 폼 제출 시 페이지 새로고침 방지

    try {
      const response = await axios.patch(`http://127.0.0.1:5000/user/noti/ALL`,
        {        access_token : access_token_with_header, update : "read"        }
      );
      console.log("response.data", response.data);
        // 서버로부터 받은 응답 처리
      if (response.data.msg === "edit related notice read success") {
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
  };

  const handleBookmarkClick = async (notification, e) => {
    setRelatedNotifications((prev) =>
      prev.map((notif) =>
        notif.id === notification.id ? { ...notif, isScrapped: !notif.isScrapped } : notif
      )
    );

    e.preventDefault();

    if(notification.isScrapped){
      try {
        const response = await axios.delete(`http://127.0.0.1:5000/user/noti/${notification.id}`,
          {        access_token : access_token_with_header      }
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
        const response = await axios.post(`http://127.0.0.1:5000/user/noti/${notification.id}`,
          {        access_token : access_token_with_header, is_scrap : 1      }
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

  const [content, setContent] = useState("");

  // 공지 검색 처리
  const filteredNotices = relatedNotifications.filter((notice) =>
    notice.title.includes(content) // 제목에 검색어 포함 여부 확인
  );

  const [isAscending, setIsAscending] = useState(true); // 오름차순/내림차순 상태

  const handleSort = () => {
    const sorted = [...relatedNotifications].sort((a, b) => {
      if (isAscending) return b.title.localeCompare(a.title); // 내림차순
        return a.title.localeCompare(b.title); // 오름차순
        })
    
    setRelatedNotifications(sorted);
    setIsAscending(!isAscending); // 정렬 상태 토글
  };

  const handleNoticeClick = async (e, notif)=>{
    e.preventDefault(); // 폼 제출 시 페이지 새로고침 방지

    try {
      const response = await axios.patch(`http://127.0.0.1:5000/user/noti/${notif.id}`,
        {        access_token : access_token_with_header, update : "read"       }
      );
      console.log("response.data", response.data);
        // 서버로부터 받은 응답 처리
      if (response.data.msg === "edit related notice read success") {
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

    navigate("/scrapNotice/detail", {state:{title: notif.title, noticeURL: notif.url, page:"scrapNotice", access_token: access_token}})

  }

  return (
    <div className="notification-app">
      {/* 상단 헤더 */}
      <Header page="Scrap" detail="detail" />

      {/* 관련 공지사항 */}
      <div className="main-content">
         {/* Sidebar */}
         <aside className="flex-none sidebar" style={{margin:"0px"}}>
          <div className="kingo-logo">
            <img src={kingoMIcon} width={"50px"} height={"75px"} alt="KINGO-M"/>
            <div className="iconTitle font-bold text-l relative bottom-3">KINGO-M</div>
          </div>
          <ul>
            <li 
                className={`menu-item ${currentView === 'main' ? 'active' : ''}`}
                onClick={() => {setCurrentView('main'); navigate("/scrapSchedule", {state:{access_token: access_token}})}}
            >
                알림 공지
            </li>
            <li 
                className={`menu-item ${currentView === 'scrap' ? 'active' : ''}`}
                onClick={() => {setCurrentView('scrap'); navigate("/scrapNotice", {state:{access_token: access_token}})}}
            >
                스크랩 공지
            </li>
            {/* <li 
                className={`menu-item ${currentView === 'trash' ? 'active' : ''}`}
                onClick={() => setCurrentView('trash')}
            >
                휴지통
            </li> */}
          </ul>
        </aside>

      <div className="related-notifications p-4" style={{margin:"0px", width:"100%"}}>

        <div className="notifications-header flex-col gap-1.5 w-full" style={{marginBottom:"0px"}}>
            {/*<h2>학사과정 조기졸업/석사과정 수업연한 단축/
            석박사통합과정 조기수료·이수포기 신청</h2>*/}
            <h2 className="w-full bolder" style={{textAlign:"left"}}>{selectedNotification.title || "선택된 공지사항 없음"}</h2>
            <div className="search-bar w-full mb-0">
              <input
                type="text"
                className="search-input"
                placeholder="검색어를 입력하세요"
                value={content} onChange={(e)=>setContent(e.target.value)}
              />
              <button className="search-button">검색</button>
            </div>
        
        </div>

        <div className="flex flex-row justify-between mb-3 sub_func w-full">
              <div className="flex-none text-xs  pt-1 cursor-pointer" onClick={handleSort}>
                  <span className="inline-block float-left" >정렬</span>
                  <img className="inline-block float-left" src={sort} width="17px" height="17px" />
              </div>
              <div className="flex-auto"></div>
              <div className="flex-none">
                  {/* <button className="alarm_del">공지 삭제</button> */}
                  <button className="mark_read justify-items-end" onClick={(e)=>handleMarkAllRead(e)}>Mark All Read</button>
              </div>
          </div>


        <ul className="notification-list">
          {(filteredNotices ? filteredNotices : relatedNotifications).map((notif) => (
            <li key={notif.id} className={`notification-item ${notif.isRead ? "read-notification" : "unread-notification"} cursor-pointer `}>
              <div className="notification-title bg-inherit"
                onClick={(e)=>handleNoticeClick(e, notif)}
              >{notif.title}</div>
              <div
                className={`scrap-notification-icon bg-inherit ${
                  notif.isScrapped ? "active2" : ""
                }`}
                onClick={(e) => handleBookmarkClick(notif, e)}
              >
                <IoBookmarkSharp className="bg-inherit"/>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
    </div>
  );
}

export default NotificationRelated;
