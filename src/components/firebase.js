import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getStorage } from "firebase/storage"

const firebaseConfig = {
  apiKey: "AIzaSyBVqIkOJRYEeijhPGvzTrRqRFn85d1DHdw",
  authDomain: "dsdfasf-fd87d.firebaseapp.com",
  databaseURL: "https://dsdfasf-fd87d-default-rtdb.firebaseio.com",
  projectId: "dsdfasf-fd87d",
  storageBucket: "dsdfasf-fd87d.appspot.com",
  messagingSenderId: "243554233185",
  appId: "1:243554233185:web:a9a7648c1aab1f34cfb320",
  measurementId: "G-5H3XZLEXYV"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
export const storage = getStorage(app);