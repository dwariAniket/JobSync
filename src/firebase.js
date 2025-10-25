import { getStorage } from "firebase/storage";
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
const firebaseConfig = {
  apiKey: "AIzaSyDe1TDoNtdJzhjuam4Z5opoynpcCMAYAps",
  authDomain: "anything-ac370.firebaseapp.com",
  projectId: "anything-ac370",
  storageBucket: "anything-ac370.firebasestorage.app",
  messagingSenderId: "623754482846",
  appId: "1:623754482846:web:d194cb92348132e54e978f",
  measurementId: "G-WHRFHQSRBN"
};
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
export const storage = getStorage(app)