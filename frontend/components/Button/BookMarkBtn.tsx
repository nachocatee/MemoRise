// 라이브러리
import axios from "axios";
import React, { useState } from "react";
import { Pressable, Image } from "react-native";

// 통신
import { BACKEND_URL } from "../../util/http";

// 타입
type BookMarkProps = {
  memoSeq: number | null;
  detailStyle: (React.CSSProperties | any)[];
  bookmarkType: boolean;
};

const BookMarkBtn: React.FC<BookMarkProps> = ({
  memoSeq,
  detailStyle,
  bookmarkType,
}) => {
  const [isBookMark, setIsBookMark] = useState<boolean>(bookmarkType);

  const changeIsBookMark = (id: number) => {
    if (isBookMark) {
      axios.delete(BACKEND_URL + `/memos/${id}/bookmarks/23`).catch((error) => {
        console.error(error);
      });
    } else {
      axios.post(BACKEND_URL + `/memos/${id}/bookmarks/23`).catch((error) => {
        console.error(error);
      });
    }
    setIsBookMark(!isBookMark);
  };

  return (
    <Pressable
      onPress={() => {
        if (memoSeq !== null) {
          changeIsBookMark(memoSeq);
        }
      }}
      style={detailStyle[0]}
    >
      {isBookMark ? (
        <Image
          source={require("../../assets/icons/bookmarkblue_fill.png")}
          style={detailStyle[1]}
        />
      ) : (
        <Image
          source={require("../../assets/icons/bookmarkblue.png")}
          style={detailStyle[1]}
        />
      )}
    </Pressable>
  );
};

export default BookMarkBtn;
