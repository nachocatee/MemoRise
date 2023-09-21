import { createSlice, PayloadAction } from "@reduxjs/toolkit";

// 타입 지정
export type UserProps = {
  nickname: string;
  email: string;
  profile_img: string;
};

const initialState: UserProps = {
  nickname: "",
  email: "",
  profile_img: "",
};

// 유저 슬라이스
const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    setNickname: (state, action: PayloadAction<string>) => {
      state.nickname = action.payload;
    },
    setEmail: (state, action: PayloadAction<string>) => {
      state.email = action.payload;
    },
    setProfileImg: (state, action: PayloadAction<string>) => {
      state.profile_img = action.payload;
    },
  },
});

export default userSlice.reducer;
export const { setNickname, setEmail, setProfileImg } = userSlice.actions;
