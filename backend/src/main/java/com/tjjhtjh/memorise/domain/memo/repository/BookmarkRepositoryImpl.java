package com.tjjhtjh.memorise.domain.memo.repository;

import com.querydsl.core.types.ExpressionUtils;
import com.querydsl.core.types.Projections;
import com.querydsl.core.types.dsl.BooleanExpression;
import com.querydsl.core.types.dsl.CaseBuilder;
import com.querydsl.jpa.JPAExpressions;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MyMemoResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;

import java.util.List;
import java.util.Optional;

import static com.tjjhtjh.memorise.domain.memo.repository.entity.QBookmark.bookmark;
import static com.tjjhtjh.memorise.domain.memo.repository.entity.QMemo.memo;

@Slf4j
public class BookmarkRepositoryImpl extends QuerydslRepositorySupport implements BookmarkRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public BookmarkRepositoryImpl(JPAQueryFactory jpaQueryFactory){
        super(Bookmark.class);
        this.queryFactory = jpaQueryFactory;
    }

    @Override
    public Optional<Bookmark> findByMemoAndUser(Long memoId, String email) {
        return queryFactory.selectFrom(bookmark)
                .where(bookmark.user.email.eq(email).and(bookmark.memo.memoSeq.eq(memoId)))
                .stream().findAny();
    }

    @Override
    public List<Bookmark> bookmarkExistCheck(Long memoId, Long userSeq) {
        return queryFactory.selectFrom(bookmark).where(bookmark.user.userSeq.eq(userSeq).and(bookmark.memo.memoSeq.eq(memoId)))
                .fetch();
    }

    @Override
    public List<MyMemoResponse> isBookmarkTrueList(Long userSeq) {
        CaseBuilder caseBuilder = new CaseBuilder();
        BooleanExpression isBookmarkedTrue = caseBuilder.when(JPAExpressions.selectOne().from(bookmark)
                        .where(bookmark.memo.memoSeq.eq(memo.memoSeq).and(bookmark.user.userSeq.eq(userSeq))).exists())
                .then(true).otherwise(false);

        return queryFactory.select(Projections.fields
                        (MyMemoResponse.class, memo.memoSeq, memo.user.nickname, memo.updatedAt, memo.content, memo.accessType, memo.file, memo.item.itemImage
                                , ExpressionUtils.as(isBookmarkedTrue, "isBookmarked")))
                .from(memo)
                .leftJoin(memo.user)
                .leftJoin(memo.item)
                .leftJoin(bookmark).on(memo.memoSeq.eq(bookmark.memo.memoSeq).and(bookmark.user.userSeq.eq(userSeq)))
                .where(memo.isDeleted.eq(0))
                .groupBy(memo.memoSeq)
                .having(isBookmarkedTrue.eq(true))
                .orderBy(memo.updatedAt.desc())
                .fetch();
    }

    @Override
    public Boolean bookmarkBoolean(Long memoId, Long userSeq) {
        Integer count =  queryFactory.selectFrom(bookmark).where(bookmark.user.userSeq.eq(userSeq).and(bookmark.memo.memoSeq.eq(memoId)))
                .fetch().size();
        return count > 0 ;
    }
}
