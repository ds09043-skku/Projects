import { useParams, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";

function NotificationDetail() {
  const access_token = localStorage.getItem('access_token');
console.log(access_token); 
  const { id } = useParams();
  const navigate = useNavigate();
  const [notification, setNotification] = useState(null);

  // 공지사항 데이터 (하드코딩)
  const notifications = [
    {
      id: 1,
      title: "2024학년도 2학기 수강신청 공지사항",
      detail: "수강신청은 2024년 8월 1일부터 시작됩니다. 본 수강신청은 학년별로 일정이 다르며, 반드시 수강신청 전 본인의 계획서를 검토하시기 바랍니다.",
    },
    {
      id: 2,
      title: "2024학년도 2학기 수강철회 안내",
      detail: "수강철회는 2024년 9월 15일까지 가능합니다. 철회 기간 이후에는 철회가 불가능하니, 신중히 결정하시기 바랍니다.",
    },
    {
      id: 3,
      title: "인공지능 특강 안내",
      detail: "이번 특강에서는 최신 AI 트렌드와 기술 응용 사례를 다룹니다. 참가를 원하는 학생들은 2024년 7월 31일까지 신청해주세요.",
    },
  ];

  // ID에 해당하는 공지사항 데이터 가져오기
  useEffect(() => {
    const fetchedNotification = notifications.find(
      (notif) => notif.id === parseInt(id)
    );
    setNotification(fetchedNotification);
  }, [id]);

  if (!notification) return <div>Loading...</div>;

  return (
    <div className="notification-detail">
      {/* 헤더 */}
      <header className="header">
        <button onClick={() => navigate(-1)}>←</button>
        <h1 className="header-title">나의 공지함</h1>
      </header>

      {/* 세부 내용 */}
      <div className="detail-content">
        <h2 className="detail-title">{notification.title}</h2>
        <p className="detail-text">{notification.detail}</p>
      </div>
    </div>
  );
}

export default NotificationDetail;

