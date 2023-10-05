package com.tjjhtjh.memorise.domain.memo.service.dto.request;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class BookmarkRequest {

    private Memo memo;
    private User user;

    public static Bookmark saveToEntity(Memo memo, User user) {
        return Bookmark.builder()
                .memo(memo)
                .user(user)
                .build();
    }

}
