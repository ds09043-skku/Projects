import logo from './logo.svg';
import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home.js';
import Schedule from './pages/Schedule.js';
import ScheduleDetail from './pages/ScheduleDetail.js';
import Chatbot from './pages/Chatbot';
import Register from './pages/Register';
import SelectedNotice from './pages/selectedNotice.js';
import NotificationApp from './pages/Scrap.js';
import ScrapNotifications from "./pages/Scrap2.js";
import NotificationDetail from "./pages/NotificationDetail";
import NotificationRelated from "./pages/Scrap3.js"
import Login from './pages/Login.js';

function App() {
  return (
    <div className="App w-screen h-screen bg-white" style={{maxWidth:'400px'}}>
      <BrowserRouter>                                   
        <Routes>                                            
          <Route path='/home' element={<Home />} />  
          <Route path='/' element={<Login />} />          
          <Route path='/schedule' element={<Schedule />} />      
          <Route path='/schedule/detail' element={<ScheduleDetail />} />      
          <Route path='/schedule/detail/notice' element={<SelectedNotice />} />      
          <Route path='/chatbot' element={<Chatbot />} />
          <Route path='/keywordRegister' element={<Register />} />
          <Route path='/keyword/notice' element={<SelectedNotice />} />
          <Route path="/scrapSchedule" element={<NotificationApp />} />
          <Route path="/scrapNotice" element={<ScrapNotifications />} />
          <Route path='/scrapNotice/detail' element={<SelectedNotice />} />      
          <Route path="/scrap/detail/:id" element={<NotificationDetail />} />
          <Route path="/srapSchedule/relatedNotice" element={<NotificationRelated />} />
          <Route path='/srapSchedule/relatedNotice/detail' element={<SelectedNotice />} />      
        </Routes>
    </BrowserRouter>
    </div>
  );
}

export default App;
