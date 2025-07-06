import Header from "../components/Header";
import "../styles/Schedule.css";

import { FaCircleCheck } from "react-icons/fa6";
import { FaStar } from "react-icons/fa";

import table from "../assets/images/table.png";
import list from "../assets/images/list.png";

import {useState, useEffect, useMemo, useRef} from "react";

import React, { Component } from 'react';
import { useNavigate, Link, useLocation} from 'react-router-dom';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import listPlugin from '@fullcalendar/list';
import dayjs from "dayjs"; // 날짜 포맷팅 라이브러리
import axios from "axios";

dayjs.locale("ko");

function Schedule(){

    // const [type, setType] = useState('Calendar');
    const [currentViewRange, setCurrentViewRange] = useState({ start: null, end: null });
    const navigate = useNavigate();
    const location = useLocation();

    const access_token = localStorage.getItem('access_token');
    console.log(access_token);
    const access_token_with_header = "Bearer " + access_token;

    const [message, setMessage] = useState("");

    const [todos, setTodos] = useState([
      {id: 1, title: '2024학년도 및 1학기 개시일', start: '2024-03-01', end: '', notice: [], new: 0, keywordid: ""},
      {id: 2, title: '1학기 개강', start: '2024-03-04', end: '', notice: [], new: 0, keywordid: ""},
      {id: 3, title: '학사과정 조기졸업/석사과정 수업연한 단축/석박사통합과정 조기수료·이수포기 신청', start: '2024-03-04', end: '2024-03-07', notice: [], new: 0, keywordid: ""},
      {id: 4, title: '대학원과정 논문제출자격시험 응시(면제) 신청', start: '2024-03-04', end: '2024-03-07', notice: [], new: 0, keywordid: ""},
      {id: 5, title: '2024학년도 1학기 추가 등록(분할납부자 포함)', start: '2024-03-04', end: '2024-03-08', notice: [], new: 0, keywordid: ""},
      {id: 6, title: '수강신청 확인/변경', start: '2024-03-04', end: '2024-03-09', notice: [], new: 0, keywordid: ""},
      {id: 7, title: '학사과정 학점포기 신청', start: '2024-03-13', end: '2024-03-15', notice: [], new: 0, keywordid: ""},
      {id: 8, title: '학생제안주간(Student Suggestion Week)', start: '2024-03-18', end: '2024-03-29', notice: [], new: 0, keywordid: ""},
      {id: 9, title: '수강철회 신청', start: '2024-03-20', end: '2024-03-22', notice: [], new: 0, keywordid: ""},
      {id: 10, title: '(학기 개시 30일)', start: '2024-03-30', end: '', notice: [], new: 0, keywordid: ""},
      {id: 11, title: '(수업일수 1/4)', start: '2024-03-31', end: '', notice: [], new: 0, keywordid: ""},
      {id: 12, title: '등록금 분할납부 신청자 2차 등록(4회 분납자)', start: '2024-04-01', end: '2024-04-03', notice: [], new: 0, keywordid: ""},
      {id: 13, title: '등록금 분할납부 신청자 최종(2회 분납자)/ 3차(4회 분납자) 등록', start: '2024-04-22', end: '2024-04-24', notice: [], new: 0, keywordid: ""},
      {id: 14, title: '1학기 중간시험', start: '2024-04-22', end: '2024-04-26', notice: [], new: 0, keywordid: ""},
      {id: 15, title: '학사과정 복수전공, 융합트랙, 마이크로디그리 1차 신청', start: '2024-04-22', end: '2024-04-26', notice: [], new: 0, keywordid: ""},
      {id: 16, title: '대학원과정 학위논문 예비·심사 신청', start: '2024-04-22', end: '2024-04-29', notice: [], new: 0, keywordid: ""},
      {id: 17, title: '1학기 중간강의평가', start: '2024-04-22', end: '2024-05-03', notice: [], new: 0, keywordid: ""},
      {id: 18, title: '(수업일수 2/4)', start: '2024-04-27', end: '', notice: [], new: 0, keywordid: ""},
      {id: 19, title: '(학기 개시 60일)', start: '2024-04-29', end: '', notice: [], new: 0, keywordid: ""},
      {id: 20, title: '등록금 분할납부 신청자 최종 등록(4회 분납자)', start: '2024-05-13', end: '2024-05-16', notice: [], new: 0, keywordid: ""},
      {id: 21, title: '(수업일수 3/4) 일반휴학신청 마감기한', start: '2024-05-25', end: '', notice: [], new: 0, keywordid: ""},
      {id: 22, title: '재학중 입대휴학자 학점인정 신청가능 입대일', start: '2024-05-26', end: '2024-06-21', notice: [], new: 0, keywordid: ""},
      {id: 23, title: '2024학년도 2학기 학사과정 학석사연계과정 신청', start: '2024-05-27', end: '2024-06-07', notice: [], new: 0, keywordid: ""},
      {id: 24, title: '(학기 개시 90일)', start: '2024-05-29', end: '', notice: [], new: 0, keywordid: ""},
      {id: 25, title: '1학기 기말강의평가', start: '2024-06-03', end: '2024-06-14', notice: [], new: 0, keywordid: ""},
      {id: 26, title: '1학기 기말시험', start: '2024-06-17', end: '2024-06-21', notice: [], new: 0, keywordid: ""},
      {id: 27, title: '1학기 성적 입력', start: '2024-06-17', end: '2024-06-27', notice: [], new: 0, keywordid: ""},
      {id: 28, title: '1학기 종강 (수업일수 4/4)', start: '2024-06-21', end: '', notice: [], new: 0, keywordid: ""},
      {id: 29, title: '여름방학', start: '2024-06-22', end: '', notice: [], new: 0, keywordid: ""},
      {id: 30, title: '여름 계절수업/ 도전학기 시작', start: '2024-06-24', end: '', notice: [], new: 0, keywordid: ""},
      {id: 31, title: '1학기 성적 공시', start: '2024-06-28', end: '2024-07-03', notice: [], new: 0, keywordid: ""},
      {id: 32, title: '2024년 8월 졸업예정 학사과정 3품인증 취득증빙 제출기한', start: '2024-06-30', end: '', notice: [], new: 0, keywordid: ""},
      {id: 33, title: '1학기 성적 확정', start: '2024-07-08', end: '', notice: [], new: 0, keywordid: ""},
      {id: 34, title: '2024년 8월 학위취득예정 대학원과정 학위논문 On-line 탑재 완료 기한', start: '2024-07-11', end: '', notice: [], new: 0, keywordid: ""},
      {id: 35, title: '학사과정 복수전공, 융합트랙, 마이크로디그리 2차 신청(교직복수전공 포함)', start: '2024-07-15', end: '2024-07-19', notice: [], new: 0, keywordid: ""},
      {id: 36, title: '2024년 8월 학위취득예정 대학원과정 학위논문 인쇄본 제출 기한', start: '2024-07-19', end: '', notice: [], new: 0, keywordid: ""},
      {id: 37, title: '2학기 재입학 신청', start: '2024-07-22', end: '2024-07-26', notice: [], new: 0, keywordid: ""},
      {id: 38, title: '2학기 복학 신청', start: '2024-07-22', end: '2024-08-02', notice: [], new: 0, keywordid: ""},
      {id: 39, title: '2학기 일반휴학 신청', start: '2024-07-29', end: '2024-09-06', notice: [], new: 0, keywordid: ""},
      {id: 40, title: '2024학년도 2학기 등록금 분할납부 신청', start: '2024-08-19', end: '2024-08-21', notice: [], new: 0, keywordid: ""},
      {id: 41, title: '2024학년도 2학기 등록/ 분할납부 신청자 1차 등록', start: '2024-08-22', end: '2024-08-29', notice: [], new: 0, keywordid: ""},
      {id: 42, title: '2024년 여름 학위수여식', start: '2024-08-23', end: '', notice: [], new: 0, keywordid: ""},
      {id: 43, title: '여름방학 종료', start: '2024-08-31', end: '', notice: [], new: 0, keywordid: ""},
      {id: 44, title: "2024학년도 2학기 개강", start: "2024-09-02", end: "", notice: [], new: 0, keywordid: ""},
      {id: 45, title: "학사과정 조기졸업/석사과정 수업연한 단축/석박사통합과정 조기수료·이수포기 신청", start: "2024-09-02", end: "2024-09-05", notice: [], new: 0, keywordid: ""},
      {id: 46, title: "대학원과정 논문제출자격시험 응시(면제) 신청", start: "2024-09-02", end: "2024-09-05", notice: [], new: 0, keywordid: ""},
      {id: 47, title: "2024학년도 2학기 추가 등록(분할납부자 포함)", start: "2024-09-02", end: "2024-09-06", notice: [], new: 0, keywordid: ""},
      {id: 48, title: "수강신청 확인/변경", start: "2024-09-02", end: "2024-09-07", notice: [], new: 0, keywordid: ""},
      {id: 49, title: "학사과정 학점포기 신청", start: "2024-09-11", end: "2024-09-13", notice: [], new: 0, keywordid: ""},
      {id: 50, title: "학생제안주간(Student Suggestion Week)", start: "2024-09-16", end: "2024-09-27", notice: [], new: 0, keywordid: ""},
      {id: 51, title: "수강철회 신청", start: "2024-09-19", end: "2024-09-21", notice: [], new: 0, keywordid: ""},
      {id: 52, title: "건학기념일", start: "2024-09-25", end: "", notice: [], new: 0, keywordid: ""},
      {id: 53, title: "공부자탄강일", start: "2024-09-28", end: "", notice: [], new: 0, keywordid: ""},
      {id: 54, title: "등록금 분할납부 신청자 2차 등록(4회 분납자)", start: "2024-10-01", end: "2024-10-04", notice: [], new: 0, keywordid: ""},
      {id: 55, title: "등록금 분할납부 신청자 최종(2회 분납자)/ 3차(4회 분납자) 등록", start: "2024-10-21", end: "2024-10-23", notice: [], new: 0, keywordid: ""},
      {id: 56, title: "2학기 중간시험", start: "2024-10-21", end: "2024-10-25", notice: [], new: 0, keywordid: ""},
      {id: 57, title: "학사과정 복수전공, 융합트랙, 마이크로디그리 1차 신청", start: "2024-10-21", end: "2024-10-25", notice: [], new: 0, keywordid: ""},
      {id: 58, title: "대학원과정 학위논문 예비·심사 신청", start: "2024-10-21", end: "2024-10-28", notice: [], new: 0, keywordid: ""},
      {id: 59, title: "2학기 중간강의평가", start: "2024-10-21", end: "2024-11-01", notice: [], new: 0, keywordid: ""},
      {id: 60, title: "학사과정 교직과정 신청", start: "2024-11-11", end: "2024-11-15", notice: [], new: 0, keywordid: ""},
      {id: 61, title: "일반휴학신청 마감기한", start: "2024-11-23", end: "", notice: [], new: 0, keywordid: ""},
      {id: 62, title: "재학중 입대휴학자 학점인정 신청가능 입대일", start: "2024-11-24", end: "2024-12-20", notice: [], new: 0, keywordid: ""},
      {id: 63, title: "2025학년도 1학기 학사과정 학석사연계과정 신청", start: "2024-11-25", end: "2024-12-06", notice: [], new: 0, keywordid: ""},
      {id: 64, title: "2학기 기말강의평가", start: "2024-12-02", end: "2024-12-13", notice: [], new: 0, keywordid: ""},
      {id: 65, title: "2학기 기말시험", start: "2024-12-16", end: "2024-12-20", notice: [], new: 0, keywordid: ""},
      {id: 66, title: "2학기 성적 입력", start: "2024-12-16", end: "2024-12-26", notice: [], new: 0, keywordid: ""},
      {id: 67, title: "2학기 종강", start: "2024-12-20", end: "", notice: [], new: 0, keywordid: ""},
      {id: 68, title: "겨울방학 시작", start: "2024-12-21", end: "", notice: [], new: 0, keywordid: ""},
      {id: 69, title: "겨울 계절수업 시작", start: "2024-12-23", end: "", notice: [], new: 0, keywordid: ""},
      {id: 70, title: "2학기 성적 공시", start: "2024-12-27", end: "2025-01-02", notice: [], new: 0, keywordid: ""},
      {id: 71, title: "2025년 2월 졸업예정 학사과정 3품인증 취득증빙 제출기한", start: "2024-12-30", end: "", notice: [], new: 0, keywordid: ""},
      {id: 72, title: "2학기 성적 확정", start: "2025-01-07", end: "", notice: [], new: 0, keywordid: ""},
      {id: 73, title: "2025년 2월 학위취득예정 대학원과정 학위논문 On-line 탑재 완료 기한", start: "2025-01-09", end: "", notice: [], new: 0, keywordid: ""},
      {id: 74, title: "학사과정 복수전공, 융합트랙, 마이크로디그리 2차 신청(교직복수전공 포함)", start: "2025-01-13", end: "2025-01-17", notice: [], new: 0, keywordid: ""},
      {id: 75, title: "2025년 2월 학위취득예정 대학원과정 학위논문 인쇄본 제출 기한", start: "2025-01-17", end: "", notice: [], new: 0, keywordid: ""},
      {id: 76, title: "2025학년도 1학기 재입학 신청", start: "2025-01-20", end: "2025-01-24", notice: [], new: 0, keywordid: ""},
      {id: 77, title: "2025학년도 1학기 복학 신청", start: "2025-01-20", end: "2025-01-31", notice: [], new: 0, keywordid: ""},
      {id: 78, title: "2025학년도 1학기 일반휴학 신청", start: "2025-01-27", end: "2025-03-07", notice: [], new: 0, keywordid: ""},
      {id: 79, title: "2025학년도 1학기 등록금 분할납부 신청", start: "2025-02-12", end: "2025-02-14", notice: [], new: 0, keywordid: ""},
      {id: 80, title: "2025학년도 1학기 등록/분할납부 신청자 1차 등록", start: "2025-02-19", end: "2025-02-27", notice: [], new: 0, keywordid: ""},
      {id: 81, title: "2025년 겨울 학위수여식", start: "2025-02-25", end: "", notice: [], new: 0, keywordid: ""},
      {id: 82, title: "겨울방학 종료", start: "2025-02-28", end: "", notice: [], new: 0, keywordid: ""},
      {id: 83, title: "2025학년도 및 1학기 개시일", start: "2025-03-01", end: "", notice: [], new: 0, keywordid: ""},
      {id: 84, title: "2025학년도 1학기 개강", start: "2025-03-04", end: "", notice: [], new: 0, keywordid: ""}
    ]);

      // 날짜가 겹치는 이벤트들을 그룹화하는 함수
      const groupEventsByDate = (events) => {
        const groupedEvents = {};
        const displayedDateRanges = new Set(); // 이미 출력된 날짜 범위를 저장

        events.forEach((event) => {
          const startDate = new Date(event.start);
          const endDate = new Date(event.end || event.start);

          const startDateString = startDate.toLocaleDateString('ko-KR');
          const endDateString = endDate.toLocaleDateString('ko-KR');

          const dateRangeKey = `${startDateString} - ${endDateString}`;

          if (!groupedEvents[dateRangeKey]) {
            groupedEvents[dateRangeKey] = { events: [], isFirstOutput: true }; // 새로운 그룹 생성
          }

          groupedEvents[dateRangeKey].events.push(event);

          // 동일한 기간이 이미 출력되었는지 확인
          if (displayedDateRanges.has(dateRangeKey)) {
            groupedEvents[dateRangeKey].isFirstOutput = false; // 표시 안 함
          } else {
            displayedDateRanges.add(dateRangeKey); // 날짜 범위를 추가
          }
        });

        // 그룹화된 이벤트들을 정렬
        const sortedGroupedEvents = Object.keys(groupedEvents)
          .sort((a, b) => {
            const [startA, endA] = a.split(' - ').map((date) => new Date(date).getTime());
            const [startB, endB] = b.split(' - ').map((date) => new Date(date).getTime());

            if (startA === startB) return endA - endB;
            return startA - startB;
          })
          .reduce((acc, key) => {
            acc[key] = groupedEvents[key];
            return acc;
          }, {});

        return sortedGroupedEvents;
      };

    const [groupedEvents, setGroupedEvents] = useState(groupEventsByDate(todos));


    useEffect(()=>{
      setGroupedEvents(groupEventsByDate(todos));
    }, [todos])

    const fullCalendarEvents = useMemo(() => {
        // console.log( "memo",groupedEvents)
        return Object.keys(groupedEvents).flatMap((key) => {
          const group = groupedEvents[key];
          // console.log("group", group)
          return group.events ? group.events.map((event, idx) => ({
            ...event,
            id: event.id,
            start: new Date(event.start),
            end: new Date(event.end),
            notice: event.notice,
            new: event.new,
            keywordid: event.keywordid,
            showDate: group.isFirstOutput || idx === 0, // 첫 번째 출력 시만 날짜 표시
          })):{};
        });
      }, [groupedEvents]);



      const transformEvents = (events) => {
        const groupedEvents = {};

        events.forEach((event) => {
          const startDate = event.start;
          const endDate = event.end
            ? event.end
            : startDate;

          const key = `${startDate}-${endDate}`;
          if (!groupedEvents[key]) {
            groupedEvents[key] = {
              ids: [],
              start: event.start,
              end: event.end,
              titles: [], // 이벤트 제목을 배열로 저장
              notices: [],
              news :[],
              keywordids: [],
              showDate : event.showDate
            };
          }
          groupedEvents[key].ids.push(event.id);
          groupedEvents[key].titles.push(event.title);
          groupedEvents[key].notices.push(event.notice);
          groupedEvents[key].news.push(event.new); // 제목 추가
          groupedEvents[key].keywordids.push(event.keywordid);
        });

        return Object.keys(groupedEvents).map((key) => {
          const group = groupedEvents[key];
          return {
            id: group.ids.length > 1 ? group.ids : group.ids[0],
            title: group.titles.join("@ "), // 여러 제목을 합쳐 표시
            start: group.start,
            end: group.end,
            notice:group.notices.length > 1 ? group.notices : group.notices[0],
            new :group.news.length > 1 ? group.news : group.news[0],
            keywordid : group.keywordids.length > 1 ? group.keywordids : group.keywordids[0],
            showDate : group.showDate
          };
        });
      };

    // const transformedEvents = useMemo(()=>transformEvents(fullCalendarEvents), [fullCalendarEvents]);

    // 현재 보이는 월을 기준으로 이벤트 필터링
    const filterEventsForCurrentViewMonth = (events) => {
        if (!currentViewRange.start || !currentViewRange.end) {
        return events; // 범위가 설정되지 않으면 모든 이벤트를 반환
        }
        // 현재 보이는 월에 해당하는 이벤트만 필터링
        return events.filter(event => {
        const eventDate = new Date(event.start);
        return eventDate >= currentViewRange.start && eventDate <= currentViewRange.end;
        });
    };

    const [currentEvents, setCurrentEvents] = useState(fullCalendarEvents);
    const [transEvents, setTransEvents] = useState(transformEvents(fullCalendarEvents));
    const [currentView, setCurrentView] = useState("dayGridMonth");
    // const [printedScheIdx, setPrintedScheIndx] = useState([]);

    // useEffect(()=>{
    //   setTransEvents(transformEvents(fullCalendarEvents));
    // },[currentEvents])

    const handleViewDidMount = (arg) => {
      // setWrittenDate([]);
      // setPrintedScheIndx(0);
      if (arg.view.type === "dayGridMonth") {
        setCurrentEvents(fullCalendarEvents);
        setCurrentView("dayGridMonth");
      } else if (arg.view.type === "listMonth") {
        setCurrentEvents(transformEvents(fullCalendarEvents));
        setCurrentView("listMonth");
      }
    };

    useEffect(()=>{
      if (currentView === "dayGridMonth") {
        setCurrentEvents(fullCalendarEvents);
      } else if (currentView === "listMonth") {
        setCurrentEvents(transformEvents(fullCalendarEvents));
      }
    },[fullCalendarEvents])

    // useEffect(()=>{
    //   console.log("currentEvents", currentEvents)
    // },[currentEvents])

    const handleDatesSet = (arg) => {
        const { view } = arg;

        setCurrentViewRange({
            start: arg.view.currentStart,
            end: arg.view.currentEnd
        });

      };

    const eventClickHandler = async (info) => {
      // info.preventDefault();

      // setTodos((prevTodos) =>
      //   prevTodos.map((todo) =>
      //     todo.id === info.event.id ? { ...todo, new: 0 } : todo
      //   )
      // );

      const { event, jsEvent } = info;

      // jsEvent를 사용하여 기본 동작을 방지
      jsEvent.preventDefault();

      console.log("click", info);

      try {
        console.log("Fetching schedule-related notices...");
        const response = await axios.post(`http://127.0.0.1:5000/user/schedule`,
          { title: info.event.title },
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

          if(response.data.count > 0)
            navigate("/schedule/detail", {state:{ title:info.event.title, notice: relatedNotice, selectedFavorites: selectedFavorites, access_token: access_token}});

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
      // if(info.event.extendedProps.notice.length > 0)
      //   navigate("/schedule/detail", {state:{ title:info.event.title, notice: info.event.extendedProps.notice, selectedFavorites: selectedFavorites, access_token: access_token}});
    };

    const eventClickHandler2 = async (event, e) => {
      // info.preventDefault();
      // setTodos((prevTodos) =>
      //   prevTodos.map((todo) =>
      //     todo.id === event.id ? { ...todo, new: 0 } : todo
      //   )
      // );

      e.preventDefault();

      try {
        const response = await axios.post(`http://127.0.0.1:5000/user/schedule`,
          { title: event.title },
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

          if(response.data.count > 0)
            navigate("/schedule/detail", {state:{ title:event.title, notice: relatedNotice, selectedFavorites: selectedFavorites, access_token: access_token}});

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
      // if(event.extendedProps.notice.length > 0)
      //   navigate("/schedule/detail", {state:{ title:event.title, notice: event.extendedProps.notice, selectedFavorites: selectedFavorites }});
    };

    const eventClickHandler3 = async (event, e) => {
      // info.preventDefault();
      console.log("click3", event);
      // setTodos((prevTodos) =>{
      //   const updatedTodos = prevTodos.map((todo) =>(
      //     todo.id === event.id ? { ...todo, new: 0 } : todo
      //   ))
      //   console.log("click3 ut",updatedTodos);
      //   return updatedTodos;
      // });

      e.preventDefault();

      try {
        console.log("Fetching schedule-related notices...");
        const response = await axios.get(`http://127.0.0.1:5000/user/schedule`, {
          headers : {Authorization: access_token_with_header}},{
          title: event.title
        });

        if (response.data.msg === "get schedule-related notices success") {
          const relatedNotice = response.data.data.map((item) => ({
            id: item.noti_id,
            title: item.title,
            url: item.url,
            read: item.read,
            isBookMarked: item.isBookMarked,
          }));

          if(response.data.count > 0)
            navigate("/schedule/detail", {state:{ title:event.title, notice: relatedNotice, selectedFavorites: selectedFavorites, access_token: access_token}});

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
      // if(event.notice.length > 0)
      //   navigate("/schedule/detail", {state:{ title:event.title, notice: event.notice, selectedFavorites: selectedFavorites }});
    };

    // const [writtenDate, setWrittenDate] = useState([]);

    // const eventDidMountHandler = (info) => {
    //   console.log("eventDidMount", info);
    //   const timeElement = info.el.querySelector(".fc-list-event-time");
    //   if (timeElement) {
    //     // 요일 정보 생성
    //     const startDay = dayjs(info.event.start).format("D일(ddd)");
    //     const endDay = info.event.end
    //       ? dayjs(info.event.end).format("D일(ddd)")
    //       : null;

    //     const eventKey = `${startDay}-${endDay}`;
    //     const isAlreadyWritten = writtenDate.includes(eventKey);

    //     // writtenDate에 기록되지 않은 경우에만 추가
    //     if (isAlreadyWritten) {
    //       timeElement.innerText = ""
    //     }else{
    //       // time 요소의 내용을 교체
    //       timeElement.innerText = `${startDay}${endDay ? ` - ${endDay}` : ""}`;
    //     }
    //   }
    // }

    // const favorites = location.state.favorites ? location.state.favorites:{};
    const [selectedFavorites, setSelectedFavorites] = useState({});

    useEffect(() => {
      // keywordid가 빈 문자열이 아닌 요소 필터링

      const fetchData = async () => {
        try {
          console.log("Fetching registered schedule...");
          const response = await axios.get(`http://127.0.0.1:5000/user/scrap`, {
            headers : {Authorization: access_token_with_header}
          });

          if (response.data.msg === "get scrap notice success") {
            const updatedTodos = todos.map((todo) => {
              const matchingData = response.data.data.find((item) => item.keyword === todo.title);
              if (matchingData) {
                setSelectedFavorites((prevState) => ({
                  ...prevState,
                  [todo.id]: true, // 해당 이벤트의 별표 상태를 반전시킴
                }));
                return {
                  ...todo,
                  keywordid: matchingData.keywordid,
                  new: matchingData.new ? 1 : 0,
                };
              }
              return todo;
            });
            setTodos(updatedTodos);

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

      fetchData();

    }, []);

    // 별표 클릭 시 상태 업데이트 함수
    const toggleFavorite = async (event, e, eventKeywordId) => {
      console.log("star togglefavor", event.id)
      setSelectedFavorites((prevState) => ({
        ...prevState,
        [event.id]: !prevState[event.id], // 해당 이벤트의 별표 상태를 반전시킴
      }));

      e.preventDefault();

      if(selectedFavorites[event.id]){
        try {
          console.log("Fetching schedule-related notices...");
          const response = await axios.delete(`http://127.0.0.1:5000/user/${eventKeywordId}`, {
            headers : {Authorization: access_token_with_header},
          });

          if (response.data.msg === "delete success") {
            setTodos((prevTodos) =>
              prevTodos.map((todo) =>
                todo.id === event.id ? { ...todo, keywordid: "" } : todo
              )
            );
            console.log("response data", response.data);
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
      else{
        try {
          console.log("Fetching schedule-related notices...");
          const response = await axios.post(`http://127.0.0.1:5000/user/keyword`,
            { keyword: event.title, is_calendar: 1 },
            { headers: { Authorization: access_token_with_header } });

          if (response.data.msg === "regist keyword success") {
            setTodos((prevTodos) =>
              prevTodos.map((todo) =>
                todo.id === event.id ? { ...todo, keywordid: response.data.keywordid } : todo
              )
            );
            console.log("response data", response.data);
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

      // // 상태 변경 후 FullCalendar 이벤트 강제 리렌더링
      // setTimeout(() => {
      //   if (calendarRef.current) {
      //     calendarRef.current.getApi().refetchEvents();
      //   }
      // }, 0);
    };

    // 리스트 일정에서 출력한 이벤트
    //   const [eventIdSet, setEventIdSet] = useState([]);

    // const handleViewChange = (view) => {
    //   // FullCalendar에서 뷰가 변경될 때마다 상태를 초기화
    //   if (view.type === 'dayGridMonth' || view.type === 'listMonth') {
    //       // setEventIdSet([]); // 상태 초기화
    //   }
    // };

    // const formatDayHeader = (dateInfo) => {
    //   const startDay = dayjs(dateInfo.start).format("D일(ddd)");
    //   const endDay = dayjs(dateInfo.end).subtract(1, "day").format("D일(ddd)");
    //   return `${startDay} - ${endDay}`;
    // };

    const printedEventIds = new Set();

    const calendarRef = useRef(null);

    // viewDidMount와 같은 역할을 하는 useEffect
    useEffect(() => {
      // 이벤트 업데이트 후 FullCalendar를 갱신
      if (calendarRef.current) {
        const calendarApi = calendarRef.current.getApi();
        calendarApi.refetchEvents();  // 이벤트 새로 고침
      }
    }, [currentEvents]);

    // useEffect(() => {
    //   if (calendarRef.current) {
    //     calendarRef.current.getApi().refetchEvents();
    //   }
    // }, [currentEvents]);

    const [currentMonth, setCurrentMonth] = useState(new Date().getMonth());

    const listViewEventContent = (arg) => {
      const { event } = arg;
      const isFavorite = selectedFavorites[arg.event.id];  // 해당 이벤트가 관심 목록에 있는지 확인

      const calendarApi = arg.view.calendar; // fullCalendar 인스턴스
      const view = calendarApi.view; // 현재 보이는 view

      // 현재 보이는 달의 범위 가져오기
      const startOfMonth = view.activeStart; // 현재 달의 시작 날짜
      const endOfMonth = view.activeEnd; // 현재 달의 끝 날짜

      // 현재 보이는 달의 이벤트 가져오기
      const eventsInCurrentMonth = calendarApi.getEvents().filter((e) => {
        return (e.start >= startOfMonth && e.start < endOfMonth) || (e.end >= startOfMonth && e.end < endOfMonth);
      });

      // 첫 번째 이벤트의 ID 가져오기
      const firstEventId = eventsInCurrentMonth.length > 0 ? eventsInCurrentMonth[0].id : null;

      console.log("First Event ID of the month:", firstEventId);


      // console.log(event.extendedProps.notice, selectedFavorites);
      // 이벤트가 여러 날짜에 걸쳐 있는지 확인하고 날짜 범위를 표시
      const startDate = event.start.toLocaleDateString('ko-KR', {
      //   month: 'numeric',
        day: 'numeric',
        weekday: 'short', // 요일 추가
      }).replace(' ', '');

      const endDate = event.end
        ? event.end.toLocaleDateString('ko-KR', {
          //   month: 'numeric',
            day: 'numeric',
            weekday: 'short', // 요일 추가
          }).replace(' ', '')
        : startDate;

      const newMonth = startOfMonth.getMonth(); // 새로 렌더링된 달의 월 (0부터 시작)

      // "다음 달"로 이동했는지 확인
      if (newMonth !== currentMonth) {
        console.log("newmonth", newMonth !== currentMonth, newMonth, currentMonth)
        printedEventIds.clear();

        setCurrentMonth(newMonth); // 새로운 월로 상태 업데이트
      }



      if (arg.view.type === 'listMonth') {
          console.log("listview",arg, currentEvents, transformEvents(fullCalendarEvents), currentEvents.toString() !== transformEvents(fullCalendarEvents).toString(), currentEvents.some(event => event.title === arg.event.title));
          const currentViewEvents = arg.view.calendar.getEvents();
          console.log("null", currentView !== 'listMonth' || currentViewEvents.toString() !== transformEvents(fullCalendarEvents).toString(), !currentEvents.some(event => event.title === arg.event.title))
          if(currentView !== 'listMonth' || currentViewEvents.toString() !== transformEvents(fullCalendarEvents).toString()){
            return null;
          }
          if(!currentEvents.some(event => event.title === arg.event.title)){
            return null;
          }
          dayjs.locale('ko');
          const startDay = dayjs(arg.event.start).format("D일(ddd)");
          const endDay = arg.event.end
            ? dayjs(arg.event.end).format("D일(ddd)")
            : null;

          // 중복 여부 확인
          const eventKey = `${startDay}${endDay ? '-'+endDay : ''} `;

          const prevStartDay = event.id > 1 ? dayjs(currentEvents[event.id-1].start).format("D일(ddd)") : "none";
          const prevEndDay = event.id > 1 ? currentEvents[event.id-1].end
            ? dayjs(currentEvents[event.id-2].end).format("D일(ddd)")
            : null : "none";

          const prevEventKey = `${prevStartDay}${prevEndDay ? '-'+prevEndDay : ''} `;
          // const isAlreadyPrinted = currentEvents[printedScheIdx] ? currentEvents[printedScheIdx].title == arg.event.title : false;

          // console.log("written", eventKey, arg.event, currentEvents[printedScheIdx], printedScheIdx, isAlreadyPrinted);
          // if (!isAlreadyPrinted) {
          //   setPrintedScheIndx((prev)=>prev+1);
          // }
          // else{
          //   return;
          // }

          const isScheInOrder = event.id.includes(',') ? event.id[0] == Number(firstEventId) + Number(printedEventIds.size) : event.id == Number(firstEventId) + Number(printedEventIds.size);

          console.log('printed',arg, event, event.id,
            printedEventIds, eventKey, currentEvents,
            printedEventIds.size, isScheInOrder, printedEventIds.has(event.id),
            arg.event.title, event.title, event.id.length, event.id[0],
            isScheInOrder, Number(firstEventId) + Number(printedEventIds.size), event.id, Number(firstEventId) + Number(printedEventIds.size) == event.id)

          if (printedEventIds.has(event.id)) {
            console.log("printed 2 true false", "return null")
            return null; // Skip rendering if already printed
          }

          // Mark the event as printed
          // if(event.id.includes(',')){
          //   var i = 0;
          //   while(i != event.id.length){
          //     printedEventIds.add(event.id[i]);
          //     i++;
          //   }
          // }
          // else{
            printedEventIds.add(event.id);
          // }

          var printedEventKeys = false;
          if(event.id > 1){
            printedEventKeys = eventKey == prevEventKey;
            console.log("printed EventKeys", eventKey == prevEventKey, eventKey,prevEventKey)

          }

          const eventLen = event.id.split(",").map(id => id.trim()).length;
          var events = []
          if(eventLen > 1 ){
            for(var i = 0; i < eventLen; i++){
              events = [...events, {id: event.id.split(",")[i],
                start: event.start,
                end: event.end,
                title:event.title.split("@")[i],
                notice: event.extendedProps.notice[i],
                new: event.extendedProps.new[i],
                keywordid: event.extendedProps.keywordid[i]
              }];
            }

          }
          console.log("events", events);

          console.log("success", event.id)

          return (
          // <Link className="w-full p-0"  to={{pathname:event.extendedProps.notice ? "/scheduleDetail":"", state:{notice : event.extendedProps.notice ? event.extendedProps.notice : "", title : event.title}}}>
          // <Link className="w-full p-0"  to={event.extendedProps.notice ? "/scheduleDetail":""} state={{notice : event.extendedProps.notice ? event.extendedProps.notice : "", title : event.title}}>
          <div className="flex" style={{minWidth:"90px", margin:"8px 0px"}}>
            <time className="flex-none my-auto pr-3" style={{textSize:"10px", minWidth:"90px", textAlign:'right'}}>
              {eventKey}
              </time>
            <div className={`flex-auto custom-event flex flex-row py-0 ${event.extendedProps.notice.length > 0 ?'cursor-pointer' : 'cursor-default'}`}>
                {event.id.split(",").map(id => id.trim()).length > 1 ?
                  <div className="flex flex-auto flex-col gap-1.5 w-full">
                  {events.map((event)=>(
                    <div key={event.id} className="flex flex-auto event-title align-middle p-1.5 m-0" style={{backgroundColor: selectedFavorites[event.id] ? 'rgb(105, 173, 1)': event.notice.length > 0 ? 'darkgray':'lightgray', borderRadius:'3px'}}>
                        <div className="flex-auto align-middle " style={{paddingTop:"2px" ,fontSize:'9px', fontWeight:"bold", backgroundColor: isFavorite ? 'rgb(105, 173, 1)' : 'inherit'}}
                         onClick={(e)=>{e.stopPropagation();eventClickHandler3(event, e)}}>
                            {event.title} <p className=" bg-inherit" style={{color:"red", display: event.new ? "inline-block" : "none"}}>new</p>
                        </div>
                        <div className="flex-none event-favorite align-middle my-auto" style={{ backgroundColor:selectedFavorites[event.id] ? 'rgb(105, 173, 1)' : "inherit"}} onClick={(e) => {e.preventDefault();e.stopPropagation();toggleFavorite(event, e, event.keywordid)}}>
                            <span className="event-favorite-star align-middle my-auto bg-inherit cursor-pointer" style={{backgroundColor:selectedFavorites[event.id] ? 'rgb(105, 173, 1)' : "inherit", color: selectedFavorites[event.id] ? 'yellow' : 'gray'} }>
                              <FaStar className="bg-inherit my-auto"/>
                            </span> {/* 별표 아이콘 */}
                        </div>
                    </div>
                  ))}
                  </div>
                  : <div className="flex flex-auto event-title align-middle p-1.5" style={{backgroundColor: isFavorite ? 'rgb(105, 173, 1)': event.extendedProps.notice.length > 0 ? 'darkgray':'lightgray', borderRadius:'3px'}}>
                    <div className="flex-auto align-middle " style={{paddingTop:"2px" ,fontSize:'9px', fontWeight:"bold", backgroundColor: isFavorite ? 'rgb(105, 173, 1)' : 'inherit'}}
                      onClick={(e)=>{e.stopPropagation();eventClickHandler2(event, e)}}
                      >
                        {event.title} <p className=" bg-inherit" style={{color:"red", display: event.extendedProps.new ? "inline-block" : "none"}}>new</p>
                    </div>
                    <div className="flex-none event-favorite align-middle my-auto z-10" style={{ backgroundColor:isFavorite ? 'rgb(105, 173, 1)' : "inherit"}} onClick={(event) => {event.stopPropagation();toggleFavorite(arg.event, event, arg.event.extendedProps.keywordid); console.log("star", isFavorite)}}>
                        <span className="event-favorite-star align-middle my-auto bg-inherit cursor-pointer" style={{backgroundColor:isFavorite ? 'rgb(105, 173, 1)' : "inherit", color: isFavorite ? 'yellow' : 'gray'} }>
                          <FaStar className="bg-inherit my-auto"/>
                        </span> {/* 별표 아이콘 */}
                    </div>
                </div> }
            </div>
          </div>
          // </Link>
      )}
      else{
          return(
          // <Link className="w-full p-0" to={{pathname:event.notice ? "/scheduleDetail":"", state:{notice : arg.event.notice ? arg.event.notice : "", title : event.title}}}>
          <div className={`flex flex-row w-full cal-custom-event ${event.extendedProps.notice.length > 0 ?'cursor-pointer' : 'cursor-default'}`} style={{backgroundColor: isFavorite ? 'rgb(105, 173, 1)' : event.extendedProps.notice.length > 0 ? 'darkgray':'lightgray'}}  >
              <div className="flex flex-auto event-title bg-inherit p-1.5" style={{fontSize:'9.5px', backgroundColor:isFavorite ? 'rgb(105, 173, 1)' : "inherit"}}
                  onClick={(e)=>{eventClickHandler2(event, e)}}
              >
                <div className="flex-none bg-inherit">{event.title} </div>
                <div className="flex-auto ml-1 bg-inherit" style={{color:"red", display: event.extendedProps.new ? "inline-block" : "none"}}>new</div>
              </div>
              <div className="flex-none event-favorite bg-inherit my-auto pt-0.5 z-10" style={{ backgroundColor:isFavorite ? 'rgb(105, 173, 1)' : "inherit"}} onClick={(event) => {event.stopPropagation();toggleFavorite(arg.event, event, arg.event.extendedProps.keywordid); }}>
                  <div className="event-favorite-star bg-inherit my-auto cursor-pointer" style={{ fontSize:"15px", color: isFavorite ? 'yellow' : 'gray', backgroundColor: isFavorite ? 'rgb(105, 173, 1)' : 'inherit'}}>
                    <FaStar className="bg-inherit my-auto" />
                  </div> {/* 별표 아이콘 */}
              </div>
          </div>
          // </Link>
        )
      }
    };

    const currentYear = new Date().getFullYear();

    const validRange = useMemo(() => ({
      start: `${currentYear}-01-01`,
      end: `${currentYear + 1}-01-01`,
    }), [currentYear]);


    return(
       <div className="relative w-full h-full bg-white">
            <Header page="Schedule"/>
            <div className="line"></div>

            {/* <div className="flex justify-between space-x-1 text-center home_tabs bg-white">
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-blue-900 id_tab cursor-pointer">신분증</button>
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-zinc-300 kingo_tab cursor-pointer">KINGO ⓘ</button>
                <button className="w-full pt-1.5 pb-2 text-sm font-semibold text-white bg-zinc-300 favor_tab cursor-pointer">즐겨찾기</button>
            </div> */}

            <div id="calender-container" className="h-fit m-3 mb-5 rounded-lg shadow-lg">

               <FullCalendar
                    ref={calendarRef}
                    className="block"
                    plugins={[listPlugin, dayGridPlugin]}
                    initialView="dayGridMonth" // 초기 View 설정
                    validRange={validRange}
                    // validRange={{
                    //   start: `2024-01-01`, // 올해 1월 1일
                    //   end: `2025-01-01`, // 내년 1월 1일
                    // }}
                    titleFormat={{ month: "long" }}
                    height="auto"
                    contentHeight={600}
                    handleWindowResize={true}
                    dragScroll={true}
                    locale="ko"
                    eventClick={eventClickHandler}
                    // eventDidMount={eventDidMountHandler}
                    listDayFormat={{
                        weekday: "short",
                        day: "numeric",
                        month: "short",
                    }} // 월/일(요일) 형식
                    listDaySideFormat={false}
                    eventContent={(arg) => listViewEventContent(arg)}
                    headerToolbar={{
                        start: "prev title next",
                        center: "",
                        end: "listMonth,dayGridMonth",
                    }}
                    buttonIcons={{
                        prev: "chevron-left",
                        next: "chevron-right",
                    }}

                    showNonCurrentDates={false}
                    fixedWeekCount={false}
                    views={{
                            dayGridMonth: {
                            dayMaxEventRows: 0,
                            buttonText: " ", // 버튼 텍스트 설정
                            dayCellContent: (arg) => {
                                return arg.date.getDate(); // 숫자만 표시
                            },
                            dayCellDidMount: (arg) => {
                                // const viewStart = arg.view.currentStart; // 현재 View의 시작일
                                // const currentMonth = viewStart.getMonth(); // 현재 월

                                // // console.log("arg",arg, currentMonth, arg.date.getMonth());
                                // // 이번 달의 날짜가 아닌 날짜들은 숨긴다.
                                // if (arg.date.getMonth() !== currentMonth) {
                                //   const startOfWeek = new Date(arg.date);
                                //   startOfWeek.setDate(arg.date.getDate() - arg.date.getDay()); // 해당 날짜의 주 첫 번째 날(일요일)을 찾음
                                //   const endOfWeek = new Date(startOfWeek);
                                //   endOfWeek.setDate(startOfWeek.getDate() + 6); // 해당 날짜의 주 마지막 날(토요일)
                                //   // console.log("arg.date",arg.date, startOfWeek, endOfWeek);

                                //   // startOfWeek가 현재 월과 같은 주에 속하는지 확인
                                //   const sameWeek = startOfWeek.getMonth() == currentMonth || endOfWeek.getMonth() == currentMonth;

                                //   console.log(startOfWeek.getMonth(), currentMonth, endOfWeek.getMonth(), sameWeek);
                                //   // 이전 달과 다음 달 날짜가 같은 주에 속하면 표시
                                //   if (arg.date.getMonth() !== currentMonth && !sameWeek) {
                                //     // arg.el.style.display = "none"; // 다른 주에 속하거나 이번 달에 속하지 않으면 숨김
                                //   }
                                // }
                            },
                        },
                        listMonth: {
                          displayEventTime:"false",
                          buttonText: " ",
                          dayHeaderContent : (arg) => {
                              // '일(요일)' 형식으로 변경
                              return new Date(arg.date).toLocaleDateString('ko-KR', {
                                  day: 'numeric',
                                  weekday: 'short',
                              });
                          }, // 버튼 텍스트 설정
                        },
                    }}
                    events={currentEvents} // 현재 보이는 월에 해당하는 이벤트만 필터링
                    // events={fullCalendarEvents}
                    datesSet={handleDatesSet} // datesSet 이벤트 핸들러 추가
                    viewDidMount={handleViewDidMount} // 뷰가 마운트될 때 호출
                    // viewWillUnmount={handleViewDidMount} // 뷰가 언마운트될 때 호출
                />

            </div>

       </div>
    )

}

export default Schedule;