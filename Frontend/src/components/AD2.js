import React from 'react';

export default function AD2({ formData, handleInputChange }) {
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
            {/* College Information Section */}
            <div style={{ marginBottom: '30px' }}>
                <div style={{ marginBottom: '20px' }}>
                    <label style={{
                        display: 'block',
                        marginBottom: '10px',
                        fontSize: '16px'
                    }}>Specialization</label>
                    <input
                        style={{
                            width: '100%',
                            padding: '10px',
                            borderRadius: '10px',
                            border: '1px solid #ddd'
                        }}
                        name='specialization'
                        value={formData.specialization}
                        onChange={handleInputChange}
                        placeholder="From which Department?"
                    />
                </div>

                <div style={{ marginBottom: '20px' }}>
                    <label style={{
                        display: 'block',
                        marginBottom: '10px',
                        fontSize: '16px'
                    }}>College Name</label>
                    <input
                        style={{
                            width: '100%',
                            padding: '10px',
                            borderRadius: '10px',
                            border: '1px solid #ddd'
                        }}
                        name='collegename'
                        value={formData.collegename}
                        onChange={handleInputChange}
                        placeholder="Enter college name (e.g., Ramakrishna Mission Shilpamandira)"
                    />
                </div>

                <div>
                    <label style={{
                        display: 'block',
                        marginBottom: '10px',
                        fontSize: '16px'
                    }}>Completion Year</label>
                    <input
                        style={{
                            width: '100%',
                            padding: '10px',
                            borderRadius: '10px',
                            border: '1px solid #ddd'
                        }}
                        name='completionyear'
                        value={formData.completionyear}
                        onChange={handleInputChange}
                        placeholder="Year of passout or expected year of passout?"
                        type="number"
                    />
                </div>
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
                            borderRadius: '1010px',
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
                    <div>
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