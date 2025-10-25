import React from 'react';

export default function AD1({ formData, handleInputChange }) {
    const [hasSkills, setHasSkills] = React.useState(null);

    const handleSkillSelection = (value) => {
        setHasSkills(value === "Yes");
        handleInputChange({ target: { name: 'haveskills', value } });
        
        if (value === "No") {
            handleInputChange({ target: { name: 'skills', value: '' } });
        }
    };

    return (
        <div style={{
            backgroundColor: 'white',
            borderRadius: '10px',
            padding: '20px',
            marginTop: '20px'
        }}>
            {/* School Medium Section */}
            <div style={{ marginBottom: '30px' }}>
                <label style={{
                    display: 'block',
                    marginBottom: '10px',
                    fontSize: '16px'
                }}>School Medium</label>
                
                <select
                    style={{
                        padding: '10px',
                        width: '100%',
                        borderRadius: '10px',
                        border: '1px solid #ddd'
                    }}
                    name='schoolmed'
                    value={formData.schoolmed}
                    onChange={handleInputChange}
                >
                    <option value="">Select Medium</option>
                    <option value="e">English</option>
                    <option value="h">Hindi</option>
                    <option value="k">Kannada</option>
                    <option value="b">Bengali</option>
                    <option value="te">Telegu</option>
                    <option value="ta">Tamil</option>
                    <option value="g">Gujrati</option>
                    <option value="mar">Marathi</option>
                    <option value="o">Odiya</option>
                    <option value="a">Assamese</option>
                    <option value="mal">Malayalam</option>
                </select>
            </div>

            {/* Skills Section */}
            <div>
                <label style={{
                    display: 'block',
                    marginBottom: '10px',
                    fontSize: '16px'
                }}>Do you have any skills in your domain?</label>
                
                <div style={{
                    display: 'flex',
                    gap: '15px',
                    marginBottom: '20px'
                }}>
                    <button
                        style={{
                            padding: '8px 15px',
                            borderRadius: '10px',
                            backgroundColor: hasSkills === true ? '#DAF7A6' : 'white',
                            border: '1px solid #ddd',
                            cursor: 'pointer',
                            flex: 1
                        }}
                        onClick={() => handleSkillSelection("Yes")}
                    >
                        Yes
                    </button>
                    <button
                        style={{
                            padding: '8px 15px',
                            borderRadius: '10px',
                            backgroundColor: hasSkills === false ? '#DAF7A6' : 'white',
                            border: '1px solid #ddd',
                            cursor: 'pointer',
                            flex: 1
                        }}
                        onClick={() => handleSkillSelection("No")}
                    >
                        No
                    </button>
                </div>

                {hasSkills && (
                    <div style={{ marginTop: '15px' }}>
                        <label style={{
                            display: 'block',
                            marginBottom: '10px'
                        }}>Mention your skills</label>
                        <input
                            style={{
                                width: '100%',
                                padding: '10px',
                                borderRadius: '10px',
                                border: '1px solid #ddd'
                            }}
                            name='skills'
                            value={formData.skills}
                            onChange={handleInputChange}
                            placeholder="List your skills separated by commas"
                        />
                    </div>
                )}
            </div>
        </div>
    );
}