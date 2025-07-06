import React, { useEffect, useState } from "react";
import "../styles/Scrap.css";
import Header from "../components/Header";
import { useNavigate, useLocation } from "react-router-dom";
import kingoMIcon from "../assets/images/kingo-M.png";
import { FaStar } from "react-icons/fa";
import sort from "../assets/images/sort.png";
import axios from "axios";



function NotificationApp() {
  const navigate = useNavigate(); // useNavigate 추가

  const location = useLocation();
  
  const access_token = localStorage.getItem('access_token');
  console.log(access_token); 
  const access_token_with_header = "Bearer " + access_token;

  const [message, setMessage] = useState("");

  const [currentView, setCurrentView] = useState("main");
  const [notifications, setNotifications] = useState([]);
  //   {
  //     id: 1,
  //     title: "학사과정 조기졸업/석사과정 수업연한 단축/석박사통합과정 조기수료·이수포기 신청",
  //     isNew: true,
  //     isRead: false,
  //     isStarred: false,
  //   },
  //   {
  //     id: 2,
  //     title: "학사과정 학점 포기 신청",
  //     isNew: false,
  //     isRead: false,
  //     isStarred: false,
  //   },
  //   {
  //     id: 3,
  //     title: "2024학년도 2학기 수강신청",
  //     isNew: true,
  //     isRead: false,
  //     isStarred: true,
  //   },
  //   {
  //     id: 4,
  //     title: "2024학년도 2학기 수강철회 안내",
  //     isNew: false,
  //     isRead: false,
  //     isStarred: false,
  //   },
  //   {
  //     id: 5,
  //     title: "인공지능",
  //     isNew: false,
  //     isRead: false,
  //     isStarred: true,
  //   },
  // ]);


  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log("Fetching keywords and schedule...");
        const response = await axios.get("http://127.0.0.1:5000/user/keyword", {
            headers: { Authorization: access_token_with_header }
        });

        if (response.data.msg === "get registerd keyword success") {
          const mappedData = response.data.data.map((item) => ({
            id: item.keywordid,
            title: item.keyword,
            isNew: item.new,
            isRead: item.read,
            isStarred: true,
          }));
          setNotifications(mappedData);
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


  

  const handleStarClick = async (id, e) => {
    setNotifications((prev) =>
      prev.map((notif) =>
        notif.id === id ? { ...notif, isStarred: !notif.isStarred } : notif
      )
    );

    e.preventDefault();

    try {
      const response = await axios.delete(`http://127.0.0.1:5000/user/${id}`,
        {   headers: { Authorization: access_token_with_header } }
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
  };

  const handleMarkAllRead = async() => {
    setNotifications((prev) =>
      prev.map((notif) => ({ ...notif, isRead: true }))
    );

    try {
      const response = await axios.patch(`http://127.0.0.1:5000/user/ALL`,
        {       headers: { Authorization: access_token_with_header }       }
      );
      console.log("response.data", response.data);
        // 서버로부터 받은 응답 처리
      if (response.data.msg === "edit regist keyword read success") {
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

  const [isAscending, setIsAscending] = useState(true); // 오름차순/내림차순 상태

  const handleSort = () => {
    const sorted = [...notifications].sort((a, b) => {
      if (isAscending) return b.title.localeCompare(a.title); // 내림차순
        return a.title.localeCompare(b.title); // 오름차순
        })
    
    setNotifications(sorted);
    setIsAscending(!isAscending); // 정렬 상태 토글
  };

  const handleNotificationClick = async (notification, e) => {
    setNotifications((prev) =>
      prev.map((notif) =>
        notif.id === notification.id ? { ...notif, isRead: true } : notif
      )
    );

    e.preventDefault(); // 폼 제출 시 페이지 새로고침 방지

    try {
      const response = await axios.patch(`http://127.0.0.1:5000/user/${notification.id}`,
        {        headers: { Authorization: access_token_with_header }     }
      );
      console.log("response.data", response.data);
        // 서버로부터 받은 응답 처리
      if (response.data.msg === "edit regist keyword read success") {
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

    try {
      const response = await axios.get(`http://127.0.0.1:5000/user/${notification.id}`,
        {       headers: { Authorization: access_token_with_header }    }
      );
      console.log("response.data", response.data);
        // 서버로부터 받은 응답 처리
      if (response.data.msg === "get regist keyword notice success") {
        console.log("response data", response.data);
        setMessage(response.data.msg); // "register keyword successful"
        if(response.data.count > 0){
          navigate(`/srapSchedule/relatedNotice`, { state: { notification : notification, access_token:access_token, keywordid : notification.id } });
        }
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

  const [content, setContent] = useState("");

  // 공지 검색 처리
  const filteredNotices = notifications.filter((notice) =>
    notice.title.includes(content) // 제목에 검색어 포함 여부 확인
  );


  return (
    <div className="notification-app w-full" style={{}}>
      <Header page="Scrap" />


      {/* Main Content */}
      <div className="main-content flex">
        {/* Sidebar */}
        <aside className="flex-none sidebar">
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

        {/* Notifications */}
        <div className="notifications flex-auto">
          <div className="notifications-header">
            <h2>알림 공지</h2>
          </div>

          <div className="search-bar">
            <input
              type="text"
              className="search-input"
              placeholder="검색어를 입력하세요"
              value={content} onChange={(e)=>setContent(e.target.value)}
            />
            <button className="search-button">검색</button>
          </div>

          <div className="flex flex-row justify-between mb-3 sub_func w-full">
              <div className="flex-none text-xs  pt-1 cursor-pointer" onClick={handleSort}>
                  <span className="inline-block float-left" >정렬</span>
                  <img className="inline-block float-left" src={sort} width="17px" height="17px" />
              </div>
              <div className="flex-auto"></div>
              <div className="flex-none">
                  {/* <button className="alarm_del">공지 삭제</button> */}
                  <button className="mark_read justify-items-end" onClick={handleMarkAllRead}>Mark All Read</button>
              </div>
          </div>
          {/* <div className="notifications-actions">
            <button onClick={handleSort} className="sort-button">
              정렬
            </button>
            <button onClick={handleMarkAllRead} className="mark-all-read">
              Mark All Read
            </button>
          </div> */}

          <ul className="notification-list">
            {(filteredNotices ? filteredNotices:notifications).map((notif) => (
              <li
                key={notif.id}
                className={`notification-item bg-inherit cursor-pointer ${
                  notif.isRead ? "read-notification" : "unread-notification"
                }`}
                style={{display:notif.isStarred ? "" :"none"}}
                onClick={(e) => handleNotificationClick(notif, e)} // onClick 이벤트 추가
              >
                <div className="notification-title bg-inherit">
                  {notif.title}
                    
                  {notif.isNew && (
                      <span className='new-label'>new</span>
                  )}
                </div>
                
                <div
                  className={`notification-icon bg-inherit ${
                    notif.isStarred ? "active" : ""
                  }`}
                  onClick={(e) => {
                    e.stopPropagation(); // 부모 onClick 이벤트 방지
                    handleStarClick(notif.id, e);
                  }}
                >
                  <FaStar className="bg-inherit"/>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default NotificationApp;
