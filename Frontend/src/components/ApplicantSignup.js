import React from 'react'
import { useNavigate } from 'react-router-dom';
export default function ApplicantSignup() {
    const navigate = useNavigate();
    const [errMsg, seterrMsg] = React.useState('');
    const [experience, setexperience] = React.useState('');
    const [fresher, setfresher] = React.useState('');
    const [fData, setfData] = React.useState({
        fname: '',
        lname: '',
        email: '',
        pw: '',
        cpw: '',
        workstatus: '',
    });

    const handlesignup = () => {
        const { fname, lname, email, pw, cpw, workstatus } = fData;
        if (fname == '' || lname == '' || email == '' || pw == '' || cpw == '' || workstatus == '') {
            seterrMsg("All fields are required");
            return;
        } if (pw != cpw) {
            seterrMsg("Password does not matched");
            return;
        }
        seterrMsg('');
        console.log(fData);
        fetch('http://192.168.29.18:4000/signup', {
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
                    alert('Registered successfully');
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

    const handleWorkStatus = (value) => {
        setfData((prev) => ({
            ...prev,
            workstatus: value,
        }));

        setexperience(value === "Experienced");
        setfresher(value === "Fresher");
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
                height: '60vh',
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
                <img src='https://st2.depositphotos.com/5390090/50482/v/450/depositphotos_504829222-stock-illustration-businessman-pencil-pen-vector-illustration.jpg' style={{
                    width: '40%',
                    height: '40%',
                    position: 'absolute',
                    top: '3vh',

                }}></img>
                <p style={{
                    position: 'absolute',
                    top: '28vh',
                    fontSize: '20px'
                }}><b>On registering, you can</b></p><br></br>
                <ul style={{
                    position: 'absolute',
                    top: '35vh',
                    listStyle: 'none',
                }}>
                    <li>&#9989;Build your profile and let recruiters recruite</li>
                    <li>&#9989;Get job posted</li>
                    <li>&#9989;Find a job and grow carrer</li>
                </ul>
            </div>

            <div style={{
                backgroundColor: 'white',
                height: '76vh',
                width: '55vw',
                display: 'flex',
                alignItems: 'center',
                padding: '15px',
                position: 'absolute',
                top: '14vh',
                left: '40vw',
                borderRadius: 10,
                display: 'flex',
                justifyContent: 'space-between',
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

                <p><b>Work status</b></p>
                <div style={{
                    display: 'flex',
                    flexDirection: 'row',
                    justifyContent: 'space-between'
                }}><button style={{
                    backgroundColor: 'white',
                    borderRadius: 10,
                    border: experience ? '2px solid #0476D0' : '1px solid #ddd',
                    height: 'auto',
                    width: '40%',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                }}
                    onClick={() => { handleWorkStatus("Experienced") }}
                ><p style={{
                    textAlign: 'left'
                }}>I'm Experienced<br></br>I have work experience excluding internships</p></button>

                    <button style={{
                        backgroundColor: 'white',
                        borderRadius: 10,
                        border: fresher ? '2px solid #0476D0' : '1px solid #ddd',
                        height: 'auto',
                        width: '40%',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                    }}
                        onClick={() => { handleWorkStatus("Fresher") }}
                    ><p style={{
                        textAlign: 'left'
                    }}>I'm Fresher<br></br>I'm a student/haven't work after study</p></button>
                </div>

            </div>

            <button style={{
                position: 'absolute',
                top: '92vh',
                left: '64vw',
                height: '5vh',
                width: '8vw',
                backgroundColor: '#0476D0',
                color: 'white',
                borderRadius: 10,
                border: '1px solid #ddd'
            }}
                onClick={() => { handlesignup() }}
            >Submit</button>

        </div>
    )
}
