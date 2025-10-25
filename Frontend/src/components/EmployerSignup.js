import React from 'react'
import { useNavigate } from 'react-router-dom';
export default function EmployerSignup() {
    const navigate = useNavigate();
    const [errMsg, seterrMsg] = React.useState('');
    const [fData, setfData] = React.useState({
        fname: '',
        lname: '',
        companyname: '',
        email: '',
        pw: '',
        cpw: '',
    });

    const handlesignup = () => {
        const { fname, lname, companyname, email, pw, cpw } = fData;
        if (fname == '' || lname == '' || companyname == '' || email == '' || pw == '' || cpw == '' ) {
            seterrMsg("All fields are required");
            return;
        } if (pw != cpw) {
            seterrMsg("Password does not matched");
            return;
        }
        seterrMsg('');
        console.log(fData);
        fetch('http://192.168.29.18:4000/employer/signup', {
            method: 'post',
            headers: {
                'content-type': 'application/json',
            },
            body: JSON.stringify(fData),
        }).then((res) => {
            if (!res.ok) {
                throw new Error(`HTTP Error! status: ${res.status}`);
            }
            return res.json();
        })
            .then((data) => {
                if (data.error) {
                    console.log(data.error);
                    seterrMsg(data.error);
                } else {
                    const employerData = {
                        name: data.fname + " " + data.lname,  // Full name
                        email: data.email,
                        companyname: data.companyname,
                    };
                    localStorage.setItem('employerData', JSON.stringify(employerData));
                    alert('Registered successfully, Now login to continue');
                    navigate('/');
                }
            })
            .catch((error) => {
                console.error('Network Error:', error);
                seterrMsg('Something went wrong. Please try again.');
            });
    }

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setfData((prev) => ({
            ...prev,
            [name]: value,
        }));
    };


    return (
        <div style={{
            height: '100vh',
            width: '100vw',
            backgroundColor: 'aliceblue',
        }}>
            <div style={{
                backgroundColor: 'white',
                height: '6vh',
                display: 'flex',
                alignItems: 'center',
                padding: '15px',
                justifyContent: 'flex-end'
            }}>
                <p style={{
                    position: 'relative',
                    top: '1vh'
                }}>Already registered? <a href='/'>Login</a> here</p>
            </div>
            <div style={{
                backgroundColor: 'white',
                height: '86vh',
                width: '30vw',
                display: 'flex',
                alignItems: 'center',
                padding: '15px',
                position: 'absolute',
                top: '14vh',
                left: '5vw',
                borderRadius: 10,
                display: 'flex',
                justifyContent: 'center',
                flexDirection: 'column',

            }}>
                <img src='https://kleistech.com/wp-content/uploads/2024/08/1699344715970.jpeg' style={{
                    width: '45%',
                    height: '30%',
                    position: 'absolute',
                    top: '3vh',

                }}></img>
                <p style={{
                    position: 'absolute',
                    top: '28vh',
                    fontSize: '13px',
                    left: '15px',
                }}><p style={{ fontSize: '25px' }}><b>Hello there!</b></p><p style={{ fontSize: '15px' }}>Welcome to Erekrut</p>evolutionizing Recruitment with Erekrut’s OneDayHire Solution Connect with Top Talent and Simplify Hiring-Hassle-Free!<br></br><br></br>Access Pre-Assessed Candidates and Streamline Your Hiring Process Empowering Recruiters with Global Talent and Campus Placements<br></br><br></br><p style={{ textDecoration: 'underline', textDecorationThickness: '1px' }}>Hire Faster, Smarter, and Better with Erekrut-The Ultimate Hiring Solution with OneDayHire</p>Disclaimer: Please be informed that any payments made outside of Erekrut's online payment gateway are solely between the involved parties, and Erekrut assumes no responsibility for such transactions.</p>

            </div>

            <div style={{
                backgroundColor: 'white',
                height: '65vh',
                width: '55vw',
                display: 'flex',
                
                padding: '15px',
                position: 'absolute',
                top: '14vh',
                left: '40vw',
                borderRadius: 10,
                display: 'flex',
                justifyContent: 'space-evenly',
                flexDirection: 'column',
            }}>{errMsg && <p style={{ color: 'red', position: 'absolute', top: '0px', left: '15px' }}>{errMsg}</p>}
                <p style={{
                    fontSize: '20px'
                }}><b>Create your profile</b></p>
                <input style={{
                    height: '7%',
                    width: '80%',
                    borderRadius: 10,
                    border: '1px solid #0476D0 ',
                    outline: 'none',
                }}
                    type='text'
                    name='fname'
                    value={fData.fname}
                    onChange={handleInputChange}
                    placeholder='Enter your first name'
                ></input>

                <input style={{
                    height: '7%',
                    width: '80%',
                    borderRadius: 10,
                    border: '1px solid #0476D0 ',
                    outline: 'none',
                }}
                    type='text'
                    name='lname'
                    value={fData.lname}
                    onChange={handleInputChange}
                    placeholder='Enter your last name'
                ></input>

                <input style={{
                    height: '7%',
                    width: '80%',
                    borderRadius: 10,
                    border: '1px solid #0476D0 ',
                    outline: 'none',
                }}
                    type='text'
                    name='lname'
                    value={fData.companyname}
                    onChange={handleInputChange}
                    placeholder='Enter company name'
                ></input>

                <input style={{
                    height: '7%',
                    width: '80%',
                    borderRadius: 10,
                    border: '1px solid #0476D0 ',
                    outline: 'none',
                }}
                    type='text'
                    name='email'
                    value={fData.email}
                    onChange={handleInputChange}
                    placeholder='Enter your Email-ID'
                ></input>

                <input style={{
                    height: '7%',
                    width: '80%',
                    borderRadius: 10,
                    border: '1px solid #0476D0 ',
                    outline: 'none',
                }}
                    type='password'
                    name='pw'
                    value={fData.pw}
                    onChange={handleInputChange}
                    placeholder='Enter your password'
                ></input>

                <input style={{
                    height: '7%',
                    width: '80%',
                    borderRadius: 10,
                    border: '1px solid #0476D0 ',
                    outline: 'none',
                }}
                    type='password'
                    name='cpw'
                    value={fData.cpw}
                    onChange={handleInputChange}
                    placeholder='confirm password'
                ></input>

            {/* <form>
  <div class="form-group">
    <label for="exampleInputEmail1">Email address</label>
    <input type="email" class="form-control" id="exampleInputEmail1" aria-describedby="emailHelp" placeholder="Enter email"/>
   
  </div>
  <div class="form-group">
    <label for="exampleInputPassword1">Password</label>
    <input type="password" class="form-control" id="exampleInputPassword1" placeholder="Password"/>
  </div>
  
  <button type="submit" class="btn btn-primary">Submit</button>
</form> */}


            </div>

            <button  style={{
                position: 'absolute',
                top: '85vh',
                left: '64vw',
                height: '5vh',
                width: '8vw',
                backgroundColor:'#0476D0',
                borderRadius:10,
                color:'white',
                border:'1px solid #0476D0'
                
            }}
                onClick={() => { handlesignup() }}
            >Submit</button>

        </div>
    )
}
