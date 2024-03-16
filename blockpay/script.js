import { initializeApp } from 'firebase/app';

const firebaseConfig = {
    apiKey: "AIzaSyCc89y5uMLg1EGiNLBDEh6Oa8xJQQIHUT4",
    authDomain: "block-pay-96e45.firebaseapp.com",
    databaseURL: "https://block-pay-96e45-default-rtdb.firebaseio.com",
    projectId: "block-pay-96e45",
    storageBucket: "block-pay-96e45.appspot.com",
    messagingSenderId: "1083689092199",
    appId: "1:1083689092199:web:ab6525ad8fcdfc1d2ac0c2"
 
};

const app = initializeApp(firebaseConfig);
const database_ref = getFirestore(app);


function register (){
    email = document.getElementById('email-form03-9').value;
    password = document.getElementById('pass-form03-9').value;
    name = document.getElementById('name-form03-9').value
}
const auth = getAuth(app);
auth.createUserWithEmailAndPassword(email,password)
.then(function (){

var user = auth.createUser


var database_ref = database_ref()
var user_data={
    email :email,
    name :full_name ,
    last_login :Data.now()

}

database_ref.child('users/'+user.uid).set(user_data)


alert('User Created')

})
.catch(function(error){
var error_code = error.error_code
var error_message = error_message

alert(error_message)
})